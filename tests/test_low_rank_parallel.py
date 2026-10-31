"""CPU math, initialization and distributed-state contracts; no GPU claims."""
import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from track_1_short.model.low_rank import LowRankParallelDown

ROOT = Path(__file__).resolve().parents[1]


def extract(relative, classes):
    """Exercise actual optimizer/tail methods without importing CUDA-only kernels."""
    tree = ast.parse((ROOT / relative).read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in classes:
            methods = classes[node.name]
            if methods is not None:
                node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
            selected.append(node)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *selected], type_ignores=[])
    env = dict(__name__=__name__, torch=torch, nn=nn, dataclass=dataclass)
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), env)
    return env


class LowRankMathTests(unittest.TestCase):
    def test_production_autograd_with_rectangular_output_and_folded_lambda(self):
        env = extract("track_1_short/perf/kernels/mlp.py", {
            "FusedLinearReLUSquareFunction": None, "FP8LowRankReadout": None})

        def quantize(src, scale, fmt):
            # Exact-arithmetic backend: tests the production custom backward's
            # algebra/shape contract, not FP8 rounding or Triton correctness.
            return src / scale, (src / scale).T.contiguous()

        def linear(a, b, aux=None, a_f8=None, b_f8=None, dequant_scale_ptr=None,
                   post_scale=None, dpre_scale=None, **kwargs):
            value = a_f8 @ b_f8.T * dequant_scale_ptr
            if aux is None:
                post = value.relu().square()
                return value, post, post / post_scale, (post / post_scale).T.contiguous()
            derivative = 2 * (aux * post_scale).sqrt() * value
            return derivative / dpre_scale, (derivative / dpre_scale).T.contiguous()

        env.update(quantize_dual_layout_fused=quantize, linear_relu_square=linear)
        mm = lambda a, b, sa, sb: (a @ b) * sa * sb
        mt = lambda a, b, sa, sb: (a @ b.T) * sa * sb
        fake_ops = SimpleNamespace(dp_f8=mm, dx_f8=mm, wg1_f8=mt, wg2_f8=mt)
        first, last = env["FusedLinearReLUSquareFunction"], env["FP8LowRankReadout"]
        one = torch.tensor(1.)
        amax = torch.zeros(1)

        def forward(x, up, a, b, p):
            row, xt = quantize(x.flatten(0, 1).detach(), one, None)
            middle = first.apply(x, up, a, up.detach(), one, row, a.detach(), p.detach(),
                                 one, amax, a.detach(), p.detach(), xt, one, up.detach(),
                                 one, one, one, amax, p)
            return last.apply(middle, b, b.detach(), b.detach(), one, one, one)

        torch.manual_seed(73)
        params = [tensor.requires_grad_() for tensor in (
            torch.randn(1, 9, 16), torch.randn(32, 16) * .1,
            torch.randn(32, 8) * .1, torch.randn(16, 8) * .1, torch.tensor(1.125))]
        cotangent = torch.randn(1, 9, 16)
        with patch.dict(globals(), linear_relu_square=linear, quantize_dual_layout_fused=quantize), \
                patch.object(torch.ops, "nanogpt", fake_ops):
            for fn in (forward, torch.compile(forward, backend="aot_eager", fullgraph=True)):
                out = fn(*params)
                x, up, a, b, p = params
                ref = (torch.relu(x @ up.T).square() @ a @ b.T) * p
                got_grads = torch.autograd.grad(out, params, cotangent)
                ref_grads = torch.autograd.grad(ref, params, cotangent)
                torch.testing.assert_close(out, ref, rtol=2e-5, atol=2e-6)
                for actual, expected in zip(got_grads, ref_grads):
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            torch._dynamo.reset()

    def test_rng_state_is_unchanged(self):
        torch.manual_seed(999)
        state = torch.get_rng_state().clone()
        first = LowRankParallelDown(32, 16, 8)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        second = LowRankParallelDown(32, 16, 8)
        torch.testing.assert_close(first.b, second.b, rtol=0, atol=0)

    def test_zero_output_can_start_learning(self):
        layer = LowRankParallelDown(32, 16, 8)
        up = torch.randn(32, 16, requires_grad=True)
        x = torch.randn(2, 5, 16, requires_grad=True)
        out = layer.reference(x, up)
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
        out.sum().backward()
        self.assertGreater(layer.a.grad.abs().sum().item(), 0)
        self.assertEqual(layer.b.grad.count_nonzero().item(), 0)
        with torch.no_grad():
            layer.a.add_(layer.a.grad, alpha=-0.001)
        layer.zero_grad()
        layer.reference(x, up).square().sum().backward()
        for grad in (layer.a.grad, layer.b.grad, up.grad, x.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)

    def test_factorized_function_and_gradients_match_dense_product(self):
        layer = LowRankParallelDown(32, 16, 8).double()
        with torch.no_grad():
            layer.a.normal_(std=0.1)
        up = torch.randn(32, 16, dtype=torch.double, requires_grad=True)
        x = torch.randn(2, 7, 16, dtype=torch.double, requires_grad=True)
        p = torch.tensor(1.13, dtype=torch.double, requires_grad=True)
        got = p * layer.reference(x, up)
        reference = torch.relu(x @ up.T).square() @ (layer.a[0] @ layer.b[0].T) * p
        cotangent = torch.randn_like(got)
        params = (x, up, layer.a, layer.b, p)
        actual = torch.autograd.grad(got, params, cotangent, retain_graph=True)
        expected = torch.autograd.grad(reference, params, cotangent)
        torch.testing.assert_close(got, reference)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def test_production_dimensions_and_reset(self):
        layer = LowRankParallelDown(2816, 768)
        self.assertEqual(layer.a.shape, (1, 2816, 384))
        self.assertEqual(layer.b.shape, (1, 768, 384))
        state = {k: v.clone() for k, v in layer.state_dict().items()}
        self.assertEqual(set(state), {"a", "b"})
        addresses = {k: v.data_ptr() for k, v in layer.named_parameters()}
        with torch.no_grad():
            layer.a.fill_(1)
            layer.b.zero_()
            layer.a_amax.fill_(5)
            layer.b_amax.fill_(6)
        layer.load_state_dict(state)
        layer.rearm()
        self.assertEqual(addresses, {k: v.data_ptr() for k, v in layer.named_parameters()})
        for k, value in layer.state_dict().items():
            torch.testing.assert_close(value, state[k])
        self.assertEqual(layer.a_amax.count_nonzero().item(), 0)
        self.assertEqual(layer.b_amax.count_nonzero().item(), 0)


class ReplicatedAnvilTests(unittest.TestCase):
    def make_optimizer(self, world_size, rank):
        env = extract("track_1_short/optim/anvil.py", {
            "ParamConfig": None,
            "AnvilAndAdam": {"_build_param_cfg", "live_bank_state", "_launch_reduce"},
        })
        dist = SimpleNamespace(all_reduce=Mock(), ReduceOp=SimpleNamespace(AVG="avg"))
        env["dist"] = dist
        optim = env["AnvilAndAdam"].__new__(env["AnvilAndAdam"])
        optim.world_size, optim.rank = world_size, rank
        optim.param_cfgs = {}
        optim.param_table = {"factor": dict(optim="anvil", comms="replicated", lr_mul=2.)}
        optim.anvil_defaults = dict(lr=.023, weight_decay=2.25, momentum=.95, beta2=.9)
        weight = nn.Parameter(torch.zeros(1, 12, 8, dtype=torch.bfloat16))
        weight.reshape = (1, 12, 8)
        optim._build_param_cfg(weight, "factor")
        optim._param_by_label = {"factor": weight}
        optim.param_states = {weight: dict(velocity=None, lane_energy=None, mantissa=None)}
        optim.banks = {"factor": SimpleNamespace(grad=torch.empty_like(weight))}
        optim._reduce_futures = {}
        return optim, weight, dist

    def test_one_and_eight_ranks_own_the_whole_factor(self):
        for world, rank in ((1, 0), (8, 0), (8, 7)):
            with self.subTest(world=world, rank=rank):
                optim, weight, dist = self.make_optimizer(world, rank)
                cfg = optim.param_cfgs[weight]
                self.assertEqual(cfg.chunk_size, 1)
                self.assertAlmostEqual(cfg.lr_mul, 2 * (12 / 8) ** 0.5)
                self.assertEqual(optim.live_bank_state("factor")["p_slice"].shape, weight.shape)
                grad = torch.randn_like(weight)
                optim._launch_reduce(weight, grad)
                torch.testing.assert_close(optim.banks["factor"].grad, grad)
                dist.all_reduce.assert_called_once_with(optim.banks["factor"].grad, op="avg", async_op=True)

    def test_tail_reads_full_replica_on_rank_seven(self):
        optim, weight, _ = self.make_optimizer(8, 7)
        env = extract("track_1_short/tail_average.py", {"TailAverages": {"_param", "_own_shard"}})
        tail = env["TailAverages"].__new__(env["TailAverages"])
        tail.optimizer, tail.rank = optim, 7
        own = tail._own_shard("factor")
        self.assertEqual(own.shape, weight.shape)
        self.assertEqual(own.data_ptr(), weight.data_ptr())


if __name__ == "__main__":
    unittest.main()

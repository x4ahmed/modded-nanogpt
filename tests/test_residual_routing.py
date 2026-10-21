"""CPU mathematical/gradient checks; GPU integration is check_rtx5090 --routing."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from track_1_short.model.residual_routing import (
    add_to_partial, append_output, block_mix, source, static_mudd, stream_total,
)


class RoutingTests(unittest.TestCase):
    def test_zero_query_is_mean_of_raw_values(self):
        a, b = torch.randn(2, 5, 8), torch.randn(2, 5, 8) * 10
        actual = block_mix((source(a),), b, torch.zeros(8))
        torch.testing.assert_close(actual, (a + b) / 2)

    def test_reference_outputs_and_gradients(self):
        torch.manual_seed(7)
        sources = [torch.randn(2, 3, 8, requires_grad=True) for _ in range(4)]
        q = torch.randn(8, requires_grad=True)
        actual = block_mix(tuple(source(s) for s in sources[:-1]), sources[-1], q)
        values = torch.stack(sources)
        keys = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
        weights = (keys * q).sum(-1).softmax(0)
        expected = (values * weights.unsqueeze(-1)).sum(0)
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(actual.square().sum(), [*sources, q], retain_graph=True)
        gb = torch.autograd.grad(expected.square().sum(), [*sources, q])
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b)

    def test_boundaries_accumulate_outputs_only(self):
        initial = torch.ones(1, 2, 8)
        for size in (1, 2, 6, 17):
            blocks, partial, site = (source(initial),), None, 0
            for _ in range(17):
                blocks, partial, site = append_output(blocks, partial, initial * 2, site, size)
            self.assertEqual(site, 17)
            self.assertEqual(len(blocks), 1 + 17 // size)
            torch.testing.assert_close(stream_total(blocks, partial), initial * 35)
            for value, scale in blocks:  # cached 1/RMS matches the block it was closed with
                torch.testing.assert_close(scale, torch.rsqrt(value.square().mean(-1) + 1e-6))

    def test_injections_do_not_move_block_boundaries(self):
        initial = torch.ones(1, 2, 8)
        blocks, partial, site = (source(initial),), None, 0
        for k in range(6):
            blocks, partial, site = append_output(blocks, partial, initial, site, 3)
            partial = add_to_partial(partial, initial * 10 if k % 2 else None)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(site, 6)
        torch.testing.assert_close(stream_total(blocks, partial), initial * (1 + 6 + 30))

    def test_no_cross_token_dependency(self):
        a, b = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
        q = torch.randn(8)
        expected = block_mix((source(a),), b, q)
        b[:, 2:] += 20
        actual = block_mix((source(a),), b, q)
        torch.testing.assert_close(actual[:, :2], expected[:, :2])

    def test_static_input_independent_and_trainable(self):
        bias = nn.Parameter(torch.arange(5, dtype=torch.float32))
        a = static_mudd(bias, 0.1, torch.randn(1, 4, 8), 5)
        b = static_mudd(bias, 0.1, torch.randn(1, 4, 8) * 20, 5)
        torch.testing.assert_close(a, b)
        a.sum().backward()
        torch.testing.assert_close(bias.grad, torch.full((5,), 0.4))

    def test_bf16_and_compiled_backward(self):
        a = torch.randn(1, 3, 8, dtype=torch.bfloat16, requires_grad=True)
        b = torch.randn_like(a, requires_grad=True)
        q = nn.Parameter(torch.zeros(8))
        compiled = torch.compile(lambda a, b, q: block_mix((source(a),), b, q), backend="aot_eager", fullgraph=True)
        actual = compiled(a, b, q)
        self.assertEqual(actual.dtype, torch.bfloat16)
        actual.float().square().sum().backward()
        for t in (a, b, q):
            self.assertTrue(torch.isfinite(t.grad).all())

    def _attnres_method(self):
        """The real _forward_attnres / _attnres_injection, with CPU stubs for the CUDA operators."""
        path = Path(__file__).resolve().parents[1] / "track_1_short/model/gpt.py"
        tree = ast.parse(path.read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GPT")
        methods = [n for n in klass.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("_forward_attnres", "_attnres_injection")]
        mixes = []
        def counted_mix(*args):
            mixes.append(args)
            return block_mix(*args)
        namespace = dict(torch=torch, F=torch.nn.functional, block_mix=counted_mix, append_output=append_output,
                         add_to_partial=add_to_partial, source=source, stream_total=stream_total,
                         norm=lambda x: torch.nn.functional.rms_norm(x, (x.shape[-1],)),
                         scale=lambda gain, x: gain * x, AttnArgs=lambda **kw: SimpleNamespace(**kw))
        # Use the source's real topology constants, in their dependency order.
        names = {"NUM_LAYERS", "NO_ATTN_LAYERS", "NO_MLP_LAYERS", "ATTN_LAYERS", "LONG_WINDOW_LAYERS",
                 "WIDE_QK_LAYERS", "PAIRED_HEAD_LAYERS", "VALUE_EMBED_GATE_CHANNELS", "POST_GATE_LAYER",
                 "ATTN_GATE_LAYERS", "PARALLEL_MLP_LAYER", "PARALLEL_MLP_SLOT", "X0_INJECT_LAYERS",
                 "BIGRAM_INJECT_LAYERS"}
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id in names
                                                   for target in node.targets for n in ast.walk(target)):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
        return namespace, mixes

    def _attnres_fake(self, namespace, inject, seen):
        def attn(h, args, *weights):
            seen.append(h)
            return h * 0.1 + (0 if args.aux_v is None else args.aux_v * 0.01)
        fake = SimpleNamespace(num_layers=11, num_heads=2, head_dim=4, ngram_dim=3, training=False,
                               attnres_block_size=6, attnres_inject=inject,
                               attnres_queries=nn.Parameter(torch.zeros(18, 8)),
                               attnres_ve_gate=nn.Parameter(torch.zeros(2, 12)),
                               attnres_last_inject=nn.Parameter(torch.tensor([0.0, 0.05])),
                               _mudd_gate_post_num_coef=13,
                               attn={str(i): attn for i in namespace["ATTN_LAYERS"]},
                               yarn=None, yarn_wide=None, yarn_paired_head=None,
                               forward_mudd_gate=lambda h, **kw: h[..., :1],
                               unpack_post_mudd_gate=lambda gate, *a: torch.sigmoid(gate),
                               _mlp=lambda h, *args: h.sin() * 0.2)
        fake._attnres_injection = lambda *a: namespace["_attnres_injection"](fake, *a)
        return fake

    def _attnres_inputs(self):
        torch.manual_seed(3)
        inputs = dict(x=torch.randn(1, 4, 8, requires_grad=True), x0=torch.randn(1, 4, 8, requires_grad=True),
                      x0_bigram=torch.randn(1, 4, 3, requires_grad=True),
                      skip_lambda=torch.tensor(0.0, requires_grad=True))
        gates = [torch.full((1, 4, 1), 0.5) for _ in range(11)]
        ve = [torch.randn(4, 8) if i in (1, 2, 8, 10) else None for i in range(11)]
        return inputs, gates, ve

    def _run_attnres(self, namespace, fake, inputs, gates, ve, f8=None, mlp_up=None):
        attn_layers = namespace["ATTN_LAYERS"]
        return namespace["_forward_attnres"](
            fake, inputs["x"], inputs["x0"], inputs["x0_bigram"], gates, gates, inputs["skip_lambda"],
            ve, [torch.zeros(2, 12) for _ in range(11)], [None]*11, [None]*11, [None]*11,
            {i: (None, None, None) for i in attn_layers}, f8, mlp_up,
            [None]*12, [None]*12, [1.0]*11, [1.0]*11,
            None, [128]*11, [False]*11, [None]*11, 2048)

    def test_real_block_forward_with_cpu_operator_stubs(self):
        for inject in (False, True):
            with self.subTest(inject=inject):
                namespace, mixes = self._attnres_method()
                seen = []
                fake = self._attnres_fake(namespace, inject, seen)
                inputs, gates, ve = self._attnres_inputs()
                output = self._run_attnres(namespace, fake, inputs, gates, ve)
                self.assertEqual(len(seen), 7)
                self.assertEqual(output.shape, inputs["x"].shape)
                # 17 sites + the readout; layer 4's gate mix is reused unless an injection changed it.
                self.assertEqual(len(mixes), 19 if inject else 18)
                output.square().sum().backward()
                self.assertGreater(fake.attnres_queries.grad.abs().sum().item(), 0)
                self.assertGreater(fake.attnres_ve_gate.grad.abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(inputs["x"].grad).all())
                last = fake.attnres_last_inject.grad
                if inject:  # layer 10's cache[0] / n-gram terms (baseline mu[10], mu[11])
                    self.assertTrue((last.abs() > 0).all())
                else:
                    self.assertIsNone(last)
                for name in ("x0", "x0_bigram", "skip_lambda"):
                    grad = inputs[name].grad
                    if inject:
                        self.assertGreater(grad.abs().sum().item(), 0, name)
                    else:
                        self.assertIsNone(grad, name)

    def test_injection_terms(self):
        namespace, _ = self._attnres_method()
        fake = self._attnres_fake(namespace, True, [])
        inputs, gates, _ = self._attnres_inputs()
        injection = namespace["_attnres_injection"](fake, 4, inputs["x0"], inputs["x0_bigram"], gates, gates)
        expected = inputs["x0"] * 0.5 + torch.nn.functional.pad(inputs["x0_bigram"] * 0.5, (0, 5))
        torch.testing.assert_close(injection, expected)
        for layer in (3, 6, 8, 10):  # the baseline's _inject adds nothing on these layers
            self.assertIsNone(namespace["_attnres_injection"](
                fake, layer, inputs["x0"], inputs["x0_bigram"], gates, gates))

    def test_fp8_control_flow_with_cpu_stubs(self):
        namespace, _ = self._attnres_method()
        namespace["quantize_dual_layout_fused"] = lambda *a, **kw: (None, None)
        fake = self._attnres_fake(namespace, True, [])
        fake.training = True
        fake._attn_x_scale = fake._attn_grad_scale = fake._mlp_x_scale = 1.0
        folds = []
        def fp8_args(slot, up, quantized, fold):
            folds.append((slot, fold))
            return ()
        fake._mlp_fp8_args = fp8_args
        inputs, gates, ve = self._attnres_inputs()
        attn_layers = namespace["ATTN_LAYERS"]
        out = self._run_attnres(namespace, fake, inputs, gates, ve,
                                f8={i: (None, None, None) for i in attn_layers}, mlp_up=[None]*12)
        out.square().sum().backward()
        self.assertIn((10, None), folds)  # last layer must not use the folded FP8 cache
        self.assertIn((11, 1.0), folds)  # parallel MLP retains its shared output gain
        self.assertEqual(len(folds), 11)
        self.assertGreater(fake.attnres_queries.grad.abs().sum().item(), 0)

    def test_static_methods_match_initial_dynamic_mudd(self):
        path = Path(__file__).resolve().parents[1] / "track_1_short/model/gpt.py"
        tree = ast.parse(path.read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GPT")
        ns = dict(torch=torch, F=torch.nn.functional, static_mudd=static_mudd, MUDD_GROUPS=12)
        for node in klass.body:
            if isinstance(node, ast.FunctionDef) and node.name in ("forward_mudd", "forward_mudd_grouped"):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
        fake = SimpleNamespace(routing="baseline", mudd_w1=torch.randn(2, 4, 8),
                               mudd_w2=torch.zeros(2, 14, 4), mudd_b2=nn.Parameter(torch.randn(2, 14)),
                               mudd_w2g=nn.Parameter(torch.zeros(14, 12, 4)), _mudd_scale=0.1,
                               mudd_static_deltas=nn.Parameter(torch.zeros(14, 12)))
        x = torch.randn(1, 3, 8)
        original = ns["forward_mudd"](fake, x, 0, 14)
        original_group, original_delta = ns["forward_mudd_grouped"](fake, x, 1, 10)
        fake.routing = "static"
        actual = ns["forward_mudd"](fake, x * 3, 0, 14)
        actual_group, actual_delta = ns["forward_mudd_grouped"](fake, x * 3, 1, 10)
        for a, b in zip(actual + actual_group, original + original_group):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(actual_delta.expand_as(original_delta), original_delta)
        (sum(t.sum() for t in actual + actual_group) + actual_delta.sum()).backward()
        self.assertGreater(fake.mudd_b2.grad.abs().sum().item(), 0)
        self.assertGreater(fake.mudd_static_deltas.grad[:10].abs().sum().item(), 0)
        self.assertIsNone(fake.mudd_w2g.grad)  # static mode no longer borrows a slice of it

    def test_optimizer_registers_both_parameters(self):
        path = Path(__file__).resolve().parents[1] / "track_1_short/training.py"
        tree = ast.parse(path.read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainingManager")
        init = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        registration = [n for n in init.body if isinstance(n, ast.If) and "model.routing" in ast.unparse(n.test)]
        table = {"ve_gate_bank": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99]},
                 "mudd_w2g": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.25},
                 "mudd_b2": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.25,
                             "wd_mul": 0.0}}
        fake = SimpleNamespace(param_table=dict(table), scatter_order=[], work_order=[])
        exec(compile(ast.Module(body=registration, type_ignores=[]), str(path), "exec"),
             dict(self=fake, model=SimpleNamespace(routing="block_attnres")))
        self.assertNotIn("mudd_static_deltas", fake.param_table)
        self.assertEqual(fake.param_table["attnres_ve_gate"], table["ve_gate_bank"])
        self.assertEqual(fake.param_table["attnres_last_inject"], table["mudd_b2"])
        static = SimpleNamespace(param_table=dict(table), scatter_order=[], work_order=[])
        exec(compile(ast.Module(body=registration, type_ignores=[]), str(path), "exec"),
             dict(self=static, model=SimpleNamespace(routing="static")))
        self.assertEqual(static.scatter_order, ["mudd_static_deltas"])
        self.assertEqual(static.param_table["mudd_static_deltas"], table["mudd_w2g"])
        for label in ("attnres_queries", "attnres_ve_gate", "attnres_last_inject"):
            self.assertIn(label, fake.scatter_order)
            self.assertIn(label, fake.work_order)
            self.assertEqual(fake.param_table[label]["optim"], "adam")
            self.assertEqual(fake.param_table[label].get("wd_mul", 1.0), 1.0 if label == "attnres_ve_gate" else 0.0)


if __name__ == "__main__":
    unittest.main()

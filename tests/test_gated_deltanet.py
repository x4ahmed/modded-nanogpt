"""CPU checks for the Gated DeltaNet layer. fla's kernel is swapped for the plain-torch recurrence, so the
custom op, its recomputing backward and the compiled trace all run here; check_rtx5090 --deltanet 1
compares the real kernel with the recurrence on the GPU."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

from track_1_short.model import gated_deltanet as gdn

ROOT = Path(__file__).resolve().parents[1]


def with_reference_kernel(test):
    return mock.patch.object(gdn, "_fla_gated_delta", gdn.gated_delta_reference)(test)


class RecurrenceTests(unittest.TestCase):
    def test_delta_rule_overwrites_a_key(self):
        # Two writes to the same key with beta = 1 and no decay: the memory holds the latest value
        # (plain linear attention would hold their sum).
        H, K, V = 1, 4, 3
        k = torch.zeros(3, H, K); k[:, 0, 0] = 1.0
        q = k.clone()
        v = torch.tensor([[[1.0, 2.0, 3.0]], [[5.0, 6.0, 7.0]], [[0.0, 0.0, 0.0]]])
        g, beta = torch.zeros(3, H), torch.tensor([[1.0], [1.0], [0.0]])
        out = gdn.gated_delta_reference(q, k, v, g, beta, torch.tensor([0, 3]))
        torch.testing.assert_close(out[1, 0], v[1, 0] / math.sqrt(K))
        torch.testing.assert_close(out[2, 0], v[1, 0] / math.sqrt(K))  # beta = 0: read only

    def test_decay_and_segment_reset(self):
        torch.manual_seed(0)
        T, H, K, V = 12, 2, 4, 3
        q, k, v = torch.randn(T, H, K), torch.randn(T, H, K), torch.randn(T, H, V)
        g, beta = -torch.rand(T, H), torch.rand(T, H)
        cu = torch.tensor([0, 5, 5, 12, 12], dtype=torch.int32)  # padded like the loader's table
        out = gdn.gated_delta_reference(q, k, v, g, beta, cu)
        alone = gdn.gated_delta_reference(q[5:], k[5:], v[5:], g[5:], beta[5:], torch.tensor([0, 7]))
        torch.testing.assert_close(out[5:], alone)

    def test_segment_offsets_and_causal_conv(self):
        cu = torch.tensor([0, 3, 3, 7, 7], dtype=torch.int32)
        self.assertEqual(gdn.segment_offsets(cu, 7).tolist(), [0, 1, 2, 0, 1, 2, 3])
        x = torch.arange(1.0, 8.0)[:, None]
        identity = torch.zeros(1, gdn.CONV_SIZE); identity[:, -1] = 1
        torch.testing.assert_close(gdn.causal_conv(x, identity, gdn.segment_offsets(cu, 7)), x)
        prev = torch.zeros(1, gdn.CONV_SIZE); prev[:, -2] = 1  # previous token only
        y = gdn.causal_conv(x, prev, gdn.segment_offsets(cu, 7))
        self.assertEqual(y[:, 0].tolist(), [0, 1, 2, 0, 4, 5, 6])  # no reach into the previous segment

    def test_init_decay(self):
        a_log, dt_bias = gdn.init_decay(6)
        dt = F.softplus(dt_bias)
        torch.testing.assert_close(dt[[0, -1]], torch.tensor([1e-3, 0.1]))
        torch.testing.assert_close(a_log.exp()[[0, -1]], torch.tensor([1.0, 16.0]))


class CustomOpTests(unittest.TestCase):
    @with_reference_kernel
    def test_custom_op_forward_backward_and_compile(self):
        torch.manual_seed(1)
        T, H, K, V = 9, 2, 4, 3
        cu = torch.tensor([0, 4, 9, 9], dtype=torch.int32)
        tensors = [torch.randn(T, H, K), torch.randn(T, H, K), torch.randn(T, H, V),
                   -torch.rand(T, H), torch.rand(T, H)]
        a = [t.clone().requires_grad_() for t in tensors]
        b = [t.clone().requires_grad_() for t in tensors]
        out = gdn.gated_delta(*a, cu)
        ref = gdn.gated_delta_reference(*b, cu)
        torch.testing.assert_close(out, ref)
        grad = torch.randn(T, H, V)
        out.backward(grad)
        ref.backward(grad)
        for x, y in zip(a, b):
            torch.testing.assert_close(x.grad, y.grad)
        compiled = torch.compile(lambda *t: gdn.gated_delta(*t, cu), backend="aot_eager", fullgraph=True)
        torch.testing.assert_close(compiled(*tensors), ref.detach())


class IntegrationTests(unittest.TestCase):
    def _gpt_namespace(self):
        path = ROOT / "track_1_short/model/gpt.py"
        tree = ast.parse(path.read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GPT")
        methods = [n for n in klass.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("init_deltanet", "_gated_delta_mixer", "attn_qk_dim", "attn_v_dim")]
        namespace = dict(torch=torch, F=F, nn=nn, CONV_SIZE=gdn.CONV_SIZE, causal_conv=gdn.causal_conv,
                         gated_delta=gdn.gated_delta, init_decay=gdn.init_decay,
                         segment_offsets=gdn.segment_offsets)
        names = {"NUM_LAYERS", "NO_ATTN_LAYERS", "ATTN_LAYERS", "LONG_WINDOW_LAYERS", "WIDE_QK_LAYERS",
                 "HALF_V_LAYERS", "NARROW_QK_LAYERS", "NARROW_HEAD_DIM"}
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id in names
                                                   for target in node.targets for n in ast.walk(target)):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
        return namespace

    @with_reference_kernel
    def test_mixer_shapes_gradients_and_rng(self):
        ns = self._gpt_namespace()
        D, H, head_dim = 16, 2, 128
        fake = nn.Module()
        fake.head_dim, fake.num_heads, fake.device, fake.deltanet_layer = head_dim, H, "cpu", 5
        for name in ("init_deltanet", "_gated_delta_mixer", "attn_qk_dim", "attn_v_dim"):
            setattr(fake, name, ns[name].__get__(fake))
        rng_before = torch.get_rng_state()
        fake.init_deltanet(D, H, head_dim)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()), "init must not draw from the global RNG")
        T = 10
        x = torch.randn(1, T, D, requires_grad=True)
        qk_w = torch.randn(2 * H * 64, D, requires_grad=True)
        v_w = torch.randn(H * 128, D, requires_grad=True)
        o_w = torch.randn(D, H * 128, requires_grad=True)
        cu = torch.tensor([0, 6, 10, 10], dtype=torch.int32)
        aux_v = torch.randn(1, T, H * head_dim)
        out = fake._gated_delta_mixer(x, cu, torch.tensor([0.5, 2.0]), torch.tensor(1.5), aux_v, qk_w, v_w, o_w)
        self.assertEqual(out.shape, (1, T, D))
        out.square().sum().backward()
        for name, p in fake.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        for t in (x, qk_w, v_w, o_w):
            self.assertGreater(t.grad.abs().sum().item(), 0)
        # Document boundary: perturbing the second segment leaves the first untouched.
        x2 = x.detach().clone()
        x2[:, 6:] += 5
        with torch.no_grad():
            a = fake._gated_delta_mixer(x.detach(), cu, torch.tensor([0.5, 2.0]), None, None, qk_w, v_w, o_w)
            b = fake._gated_delta_mixer(x2, cu, torch.tensor([0.5, 2.0]), None, None, qk_w, v_w, o_w)
        torch.testing.assert_close(a[:, :6], b[:, :6])

    def test_forward_routes_only_the_chosen_layer(self):
        gpt = (ROOT / "track_1_short/model/gpt.py").read_text()
        self.assertIn("if i == self.deltanet_layer:", gpt)
        self.assertIn("self.deltanet_layer = portable_config.DELTANET_LAYER if self.deltanet else None", gpt)

    def test_optimizer_registers_parameters(self):
        path = ROOT / "track_1_short/training.py"
        tree = ast.parse(path.read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainingManager")
        init = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        registration = [n for n in init.body if isinstance(n, ast.If) and "model.deltanet" in ast.unparse(n.test)]
        self.assertEqual(len(registration), 1)
        gpt = (ROOT / "track_1_short/model/gpt.py").read_text()
        for enabled in (False, True):
            fake = SimpleNamespace(param_table={}, scatter_order=[], work_order=[])
            exec(compile(ast.Module(body=registration, type_ignores=[]), str(path), "exec"),
                 dict(self=fake, model=SimpleNamespace(deltanet=enabled)))
            self.assertEqual(bool(fake.scatter_order), enabled)
            for label in fake.scatter_order:
                self.assertIn(f"self.{label} = nn.Parameter", gpt)  # every registered label exists


if __name__ == "__main__":
    unittest.main()

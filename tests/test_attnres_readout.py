"""CPU checks for the minimal AttnRes readout; GPU integration is check_rtx5090 --readout attnres."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from track_1_short.model.attnres_readout import X_INIT_WEIGHT, depth_readout, initial_bias

ROOT = Path(__file__).resolve().parents[1]


class ReadoutTests(unittest.TestCase):
    def test_initial_weights(self):
        torch.manual_seed(0)
        sources = [torch.randn(1, 5, 8) * (k + 1) for k in range(11)]
        out = depth_readout(sources, torch.zeros(8), initial_bias(11))
        rest = (1 - X_INIT_WEIGHT) / 10
        expected = X_INIT_WEIGHT * sources[0] + rest * sum(sources[1:])
        torch.testing.assert_close(out, expected)

    def test_reference_outputs_and_gradients(self):
        torch.manual_seed(7)
        sources = [torch.randn(2, 3, 8, requires_grad=True) for _ in range(4)]
        q = torch.randn(8, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        actual = depth_readout(sources, q, b)
        values = torch.stack(sources)
        keys = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
        weights = ((keys * q).sum(-1) + b.view(-1, 1, 1)).softmax(0)
        expected = (values * weights.unsqueeze(-1)).sum(0)
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(actual.square().sum(), [*sources, q, b], retain_graph=True)
        gb = torch.autograd.grad(expected.square().sum(), [*sources, q, b])
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y)

    def test_no_cross_token_dependency(self):
        a, c = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
        q, b = torch.randn(8), torch.zeros(2)
        expected = depth_readout((a, c), q, b)
        c[:, 2:] += 20
        torch.testing.assert_close(depth_readout((a, c), q, b)[:, :2], expected[:, :2])

    def test_bf16_and_compiled_backward(self):
        a = torch.randn(1, 3, 8, dtype=torch.bfloat16, requires_grad=True)
        c = torch.randn_like(a, requires_grad=True)
        q = nn.Parameter(torch.zeros(8))
        b = nn.Parameter(initial_bias(2))
        compiled = torch.compile(lambda a, c, q, b: depth_readout((a, c), q, b), backend="aot_eager", fullgraph=True)
        out = compiled(a, c, q, b)
        self.assertEqual(out.dtype, torch.bfloat16)
        out.float().square().sum().backward()
        for t in (a, c, q, b):
            self.assertTrue(torch.isfinite(t.grad).all())

    def test_forward_uses_readout_only_when_selected(self):
        source = (ROOT / "track_1_short/model/gpt.py").read_text()
        self.assertIn("x = depth_readout((x, *sources), self.readout_query, self.readout_bias)", source)
        self.assertIn('if self.readout == "attnres":', source)
        self.assertIn("self.readout_bias = nn.Parameter(initial_bias(11)", source)
        # The readout's 11 sources: x plus the MUDD mix's 10.
        self.assertEqual(source.count("mlp_in, late_attn_in[0],"), 1)

    def test_optimizer_registers_readout(self):
        path = ROOT / "track_1_short/training.py"
        tree = ast.parse(path.read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainingManager")
        init = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        registration = [n for n in init.body if isinstance(n, ast.If) and "model.readout" in ast.unparse(n.test)]
        self.assertEqual(len(registration), 1)
        for readout, expected in (("mudd", []), ("attnres", ["readout_query", "readout_bias"])):
            fake = SimpleNamespace(param_table={}, scatter_order=[], work_order=[])
            exec(compile(ast.Module(body=registration, type_ignores=[]), str(path), "exec"),
                 dict(self=fake, model=SimpleNamespace(readout=readout)))
            self.assertEqual(fake.scatter_order, expected)
            self.assertEqual(fake.work_order, expected)
            for label in expected:
                self.assertEqual(fake.param_table[label]["optim"], "adam")
                self.assertEqual(fake.param_table[label]["wd_mul"], 0.0)


if __name__ == "__main__":
    unittest.main()

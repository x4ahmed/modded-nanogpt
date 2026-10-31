"""CPU checks of the layer smear (model/layers.py token_shift and its wiring); no GPU needed.

GPU training remains the real test: these check the shift's arithmetic, that a zero weight leaves the
model exactly as the baseline, and that the new parameter reaches the forward and the optimizer.
"""
import ast
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from track_1_short.model.layers import token_shift  # noqa: E402


class TokenShiftTests(unittest.TestCase):
    def test_adds_a_share_of_the_previous_position(self):
        x, w = torch.randn(1, 7, 5), torch.randn(5)
        out = token_shift(x, w)
        self.assertTrue(torch.equal(out[:, 0], x[:, 0]))
        for t in range(1, 7):
            torch.testing.assert_close(out[:, t], x[:, t] + w * x[:, t - 1])

    def test_zero_weight_is_exactly_the_input(self):
        # The parameter starts at zero, so the first forward is bitwise the baseline's (bf16, signed zeros too).
        x = torch.randn(1, 64, 768).bfloat16()
        x[0, 3, :5] = -0.0
        self.assertTrue(torch.equal(token_shift(x, torch.zeros(768).bfloat16()), x))

    def test_gradients(self):
        x = torch.randn(1, 6, 4, dtype=torch.float64, requires_grad=True)
        w = torch.randn(4, dtype=torch.float64, requires_grad=True)
        g = torch.randn(1, 6, 4, dtype=torch.float64)
        token_shift(x, w).backward(g)
        torch.testing.assert_close(w.grad, (g[0, 1:] * x[0, :-1].detach()).sum(0))
        expected = g.clone()
        expected[0, :-1] += w.detach() * g[0, 1:]
        torch.testing.assert_close(x.grad, expected)
        self.assertTrue(torch.autograd.gradcheck(token_shift, (x.detach().requires_grad_(), w.detach().requires_grad_())))


class WiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpt = ast.parse((ROOT / "track_1_short/model/gpt.py").read_text(encoding="utf-8"))
        cls.training = ast.parse((ROOT / "track_1_short/training.py").read_text(encoding="utf-8"))

    def test_parameter_starts_at_zero_per_layer(self):
        init = next(n for n in ast.walk(self.gpt) if isinstance(n, ast.FunctionDef) and n.name == "init_misc")
        assigns = {ast.unparse(t): ast.unparse(n.value) for n in ast.walk(init) if isinstance(n, ast.Assign)
                   for t in n.targets}
        self.assertEqual(assigns["self.layer_smear"], "nn.Parameter(torch.zeros(num_layers, model_dim))")

    def test_every_layer_shifts_first(self):
        loop = next(n for n in ast.walk(self.gpt) if isinstance(n, ast.For)
                    and ast.unparse(n.iter) == "range(self.num_layers)" and ast.unparse(n.target) == "i")
        statements = [ast.unparse(s) for s in loop.body[:5]]
        self.assertIn("x = token_shift(x, layer_smear[i])", statements)
        # Before anything reads x in the layer (the post MUDD gate at layer 4 included).
        self.assertLess(statements.index("x = token_shift(x, layer_smear[i])"),
                        next(i for i, s in enumerate(statements) if s.startswith("if ")))

    def test_optimizer_owns_the_parameter(self):
        source = ast.unparse(self.training)
        self.assertIn("'layer_smear': {'optim': 'adam', 'comms': 'replicated', 'adam_betas': [0.9, 0.99], "
                      "'lr_mul': 1.0, 'wd_mul': 0.0}", source)
        orders = {ast.unparse(t): ast.literal_eval(n.value) for n in ast.walk(self.training)
                  if isinstance(n, ast.Assign) for t in n.targets
                  if ast.unparse(t) in ("self.scatter_order", "self.work_order")}
        for name, order in orders.items():
            with self.subTest(order=name):
                self.assertEqual(order.count("layer_smear"), 1)


if __name__ == "__main__":
    unittest.main()

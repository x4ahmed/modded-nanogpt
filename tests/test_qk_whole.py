"""CPU checks of whole-matrix Q/K whitening (optim/anvil.py merge_groups); no GPU or Triton.

The Triton Gram kernels are replaced by torch matmuls and the compiled cascade runs eagerly, so these
check the grouping bookkeeping and that only the whitening's grouping changes, not GPU numerics.
"""
import ast
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROWS, DIM, MERGE = 64, 192, 3  # 3 head-pair groups of 64 x 192 = one square Q (the model's: 3 x 256 x 768)


def load_anvil():
    """optim/anvil.py with torch stand-ins for its Triton kernels."""
    polar = types.ModuleType("track_1_short.perf.kernels.polar_express")
    polar.XXT = lambda A, out: torch.matmul(A, A.mT, out=out)
    polar.XTX = lambda A, out: torch.matmul(A.mT, A, out=out)
    polar.ba_plus_cAA = lambda A, alpha, beta, out: out.copy_(alpha * (A @ A.mT) + beta * A)
    transpose = types.ModuleType("track_1_short.perf.kernels.transpose")
    transpose.transpose_add = transpose.transpose_copy = None
    adam = types.ModuleType("track_1_short.perf.kernels.replicated_adam")
    adam.SCALAR_COLUMNS, adam.fused_adam_ = (), None
    stubs = {m.__name__: m for m in (polar, transpose, adam)}
    with mock.patch.dict(sys.modules, stubs):
        for name in ("track_1_short.optim.anvil", "track_1_short.perf.replicated_adam"):
            sys.modules.pop(name, None)
        return importlib.import_module("track_1_short.optim.anvil")


def well_conditioned(n: int) -> torch.Tensor:
    """n x n with singular values in [0.2, 1]: inside the maps' range, so whitening converges."""
    U = torch.linalg.qr(torch.randn(n, n))[0]
    V = torch.linalg.qr(torch.randn(n, n))[0]
    return U @ torch.diag(torch.linspace(1.0, 0.2, n)) @ V.T


class QKWholeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.anvil = load_anvil()
        cls.cascade = staticmethod(cls.anvil.anvil_cascade._torchdynamo_orig_callable)  # eager

    def whiten(self, grad: torch.Tensor, merge=1, velocity=None, momentum=0.0):
        """One cascade call from `velocity` (zeros by default); returns (update, velocity after)."""
        velocity = torch.zeros(2, *grad.shape) if velocity is None else velocity.clone()
        scalar = lambda v: torch.tensor(v, dtype=torch.float32)
        out = self.cascade(grad.clone(), velocity, scalar(momentum), split_baddbmm=False,
                           fast_beta_t=scalar(0.9), fast_weight_t=scalar(1.0), merge_groups=merge)
        return out, velocity

    def test_merged_equals_whitening_each_whole_matrix_alone(self):
        grad = torch.randn(2 * MERGE, ROWS, DIM).bfloat16()  # one layer's Q groups, then its K groups
        merged, _ = self.whiten(grad, MERGE)
        self.assertEqual(merged.shape, grad.shape)
        # The unchanged per-matrix path, run on a bank whose matrices are the whole Q and K.
        alone, _ = self.whiten(grad.reshape(2, MERGE * ROWS, DIM))
        self.assertTrue(torch.equal(merged.reshape(2, MERGE * ROWS, DIM), alone))

    def test_whole_matrices_are_whitened(self):
        grad = torch.stack([well_conditioned(DIM) for _ in range(2)]).reshape(2 * MERGE, ROWS, DIM)
        merged, _ = self.whiten(grad.bfloat16(), MERGE)
        sv = torch.linalg.svdvals(merged.reshape(2, MERGE * ROWS, DIM).float())
        self.assertLess((sv - 1).abs().max().item(), 0.05)

    def test_groups_of_one_matrix_become_orthogonal_to_each_other(self):
        # What changes: per group, two head pairs with the same gradient direction keep the same update
        # direction; whitening the whole Q makes every group's rows orthogonal to the other groups'.
        grad = well_conditioned(DIM).reshape(MERGE, ROWS, DIM).repeat(2, 1, 1)
        grad[1] = grad[0] + 0.1 * torch.randn(ROWS, DIM) / DIM ** 0.5
        overlap = lambda X: torch.diagonal(X[0] @ X[1].T).mean().item()
        merged, _ = self.whiten(grad.bfloat16(), MERGE)
        per_group, _ = self.whiten(grad.bfloat16())
        self.assertLess(abs(overlap(merged.float())), 0.2)
        self.assertGreater(overlap(per_group.float()), 0.8)

    def test_momentum_state_does_not_depend_on_grouping(self):
        grad = torch.randn(2 * MERGE, ROWS, DIM).bfloat16()
        velocity = torch.randn(2, 2 * MERGE, ROWS, DIM)
        _, v_merged = self.whiten(grad, MERGE, velocity, momentum=0.9)
        _, v_groups = self.whiten(grad, 1, velocity, momentum=0.9)
        self.assertTrue(torch.equal(v_merged, v_groups))

    def test_unused_head_rows_stay_zero(self):
        # Narrow layers read only the leading half of each head's Q/K rows (2 heads per group here).
        grad = torch.randn(2 * MERGE, 2, ROWS // 2, DIM)
        grad[:, :, ROWS // 4:] = 0
        merged, _ = self.whiten(grad.reshape(2 * MERGE, ROWS, DIM).bfloat16(), MERGE)
        heads = merged.reshape(2 * MERGE, 2, ROWS // 2, DIM)
        self.assertEqual(heads[:, :, ROWS // 4:].abs().max().item(), 0)
        self.assertGreater(heads[:, :, :ROWS // 4].abs().max().item(), 0)


class WiringTests(unittest.TestCase):
    def test_only_qk_bank_is_merged(self):
        tree = ast.parse((ROOT / "track_1_short/model/gpt.py").read_text(encoding="utf-8"))
        marks = [ast.unparse(t) for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets
                 if isinstance(t, ast.Attribute) and t.attr == "whiten_merge"]
        self.assertEqual(marks, ["self.qk_bank.whiten_merge"])

    def test_bank_update_passes_the_merge(self):
        tree = ast.parse((ROOT / "track_1_short/optim/anvil.py").read_text(encoding="utf-8"))
        update = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "anvil_bank_update")
        call = next(n for n in ast.walk(update) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "anvil_cascade")
        self.assertEqual({k.arg: ast.unparse(k.value) for k in call.keywords}.get("merge_groups"), "bank.whiten_merge")


if __name__ == "__main__":
    unittest.main()

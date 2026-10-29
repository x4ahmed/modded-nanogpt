"""CPU checks of per-head V/O whitening (optim/anvil.py); no GPU or Triton.

The Triton Gram kernels are replaced by torch matmuls and the compiled cascade runs eagerly, so these
check the head-block bookkeeping and that only the whitening's grouping changes, not GPU numerics.
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

DIM, HEAD_DIM = 256, 64  # square V/O matrices of H = 4 heads (the model's are 768 = 6 x 128)
H = DIM // HEAD_DIM


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


class VOHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.anvil = load_anvil()
        cls.cascade = staticmethod(cls.anvil.anvil_cascade._torchdynamo_orig_callable)  # eager

    def whiten(self, grad: torch.Tensor, head_dim=None, velocity=None, momentum=0.0):
        """One cascade call from `velocity` (zeros by default); returns (update, velocity after)."""
        velocity = torch.zeros(2, *grad.shape) if velocity is None else velocity.clone()
        scalar = lambda v: torch.tensor(v, dtype=torch.float32)
        out = self.cascade(grad.clone(), velocity, scalar(momentum), split_baddbmm=False,
                           fast_beta_t=scalar(0.9), fast_weight_t=scalar(1.0), head_dim=head_dim)
        return out, velocity

    def test_blocks_are_v_rows_and_transposed_o_columns(self):
        X = torch.randn(4, DIM, DIM)
        blocks = self.anvil.vo_head_blocks(X, HEAD_DIM)
        self.assertEqual(tuple(blocks.shape), (4 * H, HEAD_DIM, DIM))
        for m in range(4):
            for h in range(H):
                head = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
                want = X[m][head] if m % 2 == 0 else X[m][:, head].T  # V at even indices, O at odd
                self.assertTrue(torch.equal(blocks[m * H + h], want), (m, h))
        self.assertTrue(torch.equal(self.anvil.vo_from_head_blocks(blocks, HEAD_DIM), X))

    def test_per_head_equals_whitening_each_head_alone(self):
        grad = torch.randn(4, DIM, DIM).bfloat16()
        per_head, _ = self.whiten(grad, HEAD_DIM)
        # The unchanged whole-matrix path, run on a bank whose matrices are the head blocks.
        alone, _ = self.whiten(self.anvil.vo_head_blocks(grad, HEAD_DIM))
        self.assertTrue(torch.equal(self.anvil.vo_head_blocks(per_head, HEAD_DIM), alone))

    def test_every_head_block_is_whitened(self):
        grad = torch.randn(4, DIM, DIM).bfloat16()
        per_head, _ = self.whiten(grad, HEAD_DIM)
        sv = torch.linalg.svdvals(self.anvil.vo_head_blocks(per_head, HEAD_DIM).float())
        self.assertLess((sv - 1).abs().max().item(), 0.05)

    def test_heads_are_no_longer_forced_orthogonal_to_each_other(self):
        # What changes: whitening a whole V makes every head's rows orthogonal to every other head's;
        # per head, two heads with the same gradient direction keep the same update direction.
        grad = torch.randn(4, DIM, DIM)
        grad[0, HEAD_DIM:2 * HEAD_DIM] = grad[0, :HEAD_DIM] + 0.1 * torch.randn(HEAD_DIM, DIM)
        overlap = lambda V: torch.diagonal(V[:HEAD_DIM] @ V[HEAD_DIM:2 * HEAD_DIM].T).mean().item()
        per_head, _ = self.whiten(grad.bfloat16(), HEAD_DIM)
        whole, _ = self.whiten(grad.bfloat16())
        self.assertGreater(overlap(per_head[0].float()), 0.8)
        self.assertLess(abs(overlap(whole[0].float())), 0.2)

    def test_momentum_state_does_not_depend_on_grouping(self):
        grad = torch.randn(4, DIM, DIM).bfloat16()
        velocity = torch.randn(2, 4, DIM, DIM)
        _, v_heads = self.whiten(grad, HEAD_DIM, velocity, momentum=0.9)
        _, v_whole = self.whiten(grad, None, velocity, momentum=0.9)
        self.assertTrue(torch.equal(v_heads, v_whole))

    def test_half_width_heads_keep_their_zero_rows(self):
        # Layers 1 and 8 read only the leading half of each head's V rows and O columns.
        grad = torch.randn(4, DIM, DIM)
        for m in range(4):
            heads = grad[m].view(H, HEAD_DIM, DIM) if m % 2 == 0 else grad[m].view(DIM, H, HEAD_DIM).permute(1, 2, 0)
            heads[:, HEAD_DIM // 2:] = 0
        per_head, _ = self.whiten(grad.bfloat16(), HEAD_DIM)
        blocks = self.anvil.vo_head_blocks(per_head, HEAD_DIM).float()
        self.assertEqual(blocks[:, HEAD_DIM // 2:].abs().max().item(), 0)
        live = torch.linalg.svdvals(blocks[:, :HEAD_DIM // 2])
        self.assertLess((live - 1).abs().max().item(), 0.05)


class WiringTests(unittest.TestCase):
    def test_only_vo_bank_is_whitened_per_head(self):
        tree = ast.parse((ROOT / "track_1_short/model/gpt.py").read_text(encoding="utf-8"))
        marks = [ast.unparse(t) for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets
                 if isinstance(t, ast.Attribute) and t.attr == "whiten_head_dim"]
        self.assertEqual(marks, ["self.vo_bank.whiten_head_dim"])

    def test_bank_update_passes_the_head_dim(self):
        tree = ast.parse((ROOT / "track_1_short/optim/anvil.py").read_text(encoding="utf-8"))
        update = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "anvil_bank_update")
        call = next(n for n in ast.walk(update) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "anvil_cascade")
        self.assertEqual({k.arg: ast.unparse(k.value) for k in call.keywords}.get("head_dim"), "bank.whiten_head_dim")


if __name__ == "__main__":
    unittest.main()

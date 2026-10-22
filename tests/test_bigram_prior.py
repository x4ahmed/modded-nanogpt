"""CPU checks for the bigram output prior; GPU integration is check_rtx5090 --bigram-prior 1."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from track_1_short.bigram_prior import add_prior_, beta_grad, build_bigram_prior, prior_columns

ROOT = Path(__file__).resolve().parents[1]


def reference_prior(tokens, vocab, k, kappa):
    """Plain-Python version of build_bigram_prior."""
    tokens = [int(t) for t in tokens]
    pairs, count_a, count_b = {}, [0] * vocab, [0] * vocab
    for a, b in zip(tokens[:-1], tokens[1:]):
        pairs[(a, b)] = pairs.get((a, b), 0) + 1
        count_a[a] += 1
        count_b[b] += 1
    n = len(tokens) - 1
    p = [(c + 1) / (n + vocab) for c in count_b]
    table = {}
    for a in range(vocab):
        conts = sorted(((c, b) for (x, b), c in pairs.items() if x == a and b != 0), key=lambda t: (-t[0], t[1]))
        table[a] = [(b, math.log(c + kappa * p[b]) - math.log(count_a[a] + kappa) - math.log(p[b]))
                    for c, b in conts[:k]]
    return table


class BigramPriorTests(unittest.TestCase):
    def test_matches_reference(self):
        torch.manual_seed(0)
        vocab, k, kappa = 12, 3, 5.0
        tokens = torch.randint(0, vocab, (400,))
        idx, val = build_bigram_prior(tokens, vocab, k, kappa)
        ref = reference_prior(tokens, vocab, k, kappa)
        for a in range(vocab):
            got = [(int(b), float(v)) for b, v in zip(idx[a], val[a]) if b > 0]
            self.assertEqual(len(got), len(ref[a]))
            # Ties in count may order differently; compare as sets of (class, value).
            for (b, v) in got:
                match = [rv for rb, rv in ref[a] if rb == b]
                self.assertEqual(len(match), 1, (a, b))
                self.assertAlmostEqual(v, match[0], places=4)
            counts = {}
            for x, y in zip(tokens[:-1].tolist(), tokens[1:].tolist()):
                if x == a and y != 0:
                    counts[y] = counts.get(y, 0) + 1
            kept = sorted((counts[b] for b, _ in got), reverse=True)
            self.assertEqual(kept, sorted(counts.values(), reverse=True)[:len(kept)])

    def test_empty_slots_and_class_zero(self):
        tokens = torch.tensor([1, 0, 1, 2, 1, 2])
        idx, val = build_bigram_prior(tokens, 4, 4, 10.0)
        self.assertTrue((idx[1] != 0).sum() == 1)  # 1->0 is dropped, 1->2 kept
        self.assertEqual(int(idx[1, 0]), 2)
        self.assertEqual(float(val[3].abs().sum()), 0.0)  # never-seen previous token: empty row

    def test_add_prior_float_and_fp8(self):
        torch.manual_seed(1)
        idx = torch.tensor([[2, 3, 0], [1, 0, 0]], dtype=torch.int32)
        val = torch.tensor([[1.5, -0.5, 9.0], [2.0, 9.0, 9.0]])
        prev = torch.tensor([0, 1, 0])
        cols, vals = prior_columns(idx, val, prev)
        self.assertEqual(cols.tolist(), [[2, 3, 0], [1, 0, 0], [2, 3, 0]])
        self.assertEqual(vals[1].tolist(), [2.0, 0.0, 0.0])
        beta = torch.tensor(0.5)
        base = torch.randn(3, 6)
        expected = base.clone()
        for r in range(3):
            for c, v in zip(cols[r].tolist(), vals[r].tolist()):
                if v:
                    expected[r, c] += 0.5 * v
        torch.testing.assert_close(add_prior_(base.clone(), cols, vals, beta), expected)
        f8 = base.to(torch.float8_e4m3fn)
        out = add_prior_(f8.clone(), cols, vals, beta).float()
        ref = (f8.float() + (expected - base)).to(torch.float8_e4m3fn).float()
        torch.testing.assert_close(out, ref)

    def test_fp8_path_traces_under_compile(self):
        idx = torch.tensor([[2, 3], [1, 0]], dtype=torch.int32)
        val = torch.tensor([[1.0, 2.0], [3.0, 0.0]])

        def step(logits, prev, beta):
            cols, vals = prior_columns(idx, val, prev)
            return add_prior_(logits, cols, vals, beta).float()

        logits = torch.randn(2, 5).to(torch.float8_e4m3fn)
        prev, beta = torch.tensor([0, 1]), torch.tensor(1.0)
        eager = step(logits.clone(), prev, beta)
        compiled = torch.compile(step, backend="aot_eager", fullgraph=True)(logits.clone(), prev, beta)
        torch.testing.assert_close(compiled, eager)

    def test_sampled_columns_skip_absent_and_position_zero(self):
        idx = torch.tensor([[5, 7, 9]], dtype=torch.int32)
        val = torch.ones(1, 3)
        vocab_pos = torch.full((10,), -1, dtype=torch.int32)
        vocab_pos[5], vocab_pos[9] = 0, 4  # class 5 sits at candidate position 0: no prior there
        cols, vals = prior_columns(idx, val, torch.tensor([0]), vocab_pos)
        self.assertEqual(cols.tolist(), [[0, 0, 4]])
        self.assertEqual(vals.tolist(), [[0.0, 0.0, 1.0]])

    def test_beta_grad_matches_autograd(self):
        torch.manual_seed(2)
        logits = torch.randn(4, 8)
        target = torch.randint(0, 8, (4,))
        idx = torch.randint(1, 8, (8, 2), dtype=torch.int32)
        idx[:, 1] = torch.where(idx[:, 1] == idx[:, 0], 0, idx[:, 1])
        val = torch.randn(8, 2)
        cols, vals = prior_columns(idx, val, torch.randint(0, 8, (4,)))
        beta = torch.tensor(0.7, requires_grad=True)
        delta = torch.zeros(4, 8).scatter_add(1, cols, vals)
        loss = torch.nn.functional.cross_entropy(logits + beta * delta, target, reduction="sum")
        loss.backward()
        with torch.no_grad():
            z = logits + 0.7 * delta
            grad_logits = torch.softmax(z, -1) - torch.nn.functional.one_hot(target, 8)
        grad_s = 2.0 ** -6
        g8 = (grad_logits / grad_s).to(torch.float8_e5m2)
        approx = beta_grad(g8, cols, vals, grad_s)
        torch.testing.assert_close(approx, beta.grad, atol=0.05, rtol=0.1)

    def test_optimizer_registers_beta(self):
        path = ROOT / "track_1_short/training.py"
        tree = ast.parse(path.read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainingManager")
        init = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        registration = [n for n in init.body if isinstance(n, ast.If) and "model.bigram_prior" in ast.unparse(n.test)]
        self.assertEqual(len(registration), 1)
        for enabled, expected in ((False, []), (True, ["bigram_prior_beta"])):
            fake = SimpleNamespace(param_table={}, scatter_order=[], work_order=[])
            exec(compile(ast.Module(body=registration, type_ignores=[]), str(path), "exec"),
                 dict(self=fake, model=SimpleNamespace(bigram_prior=enabled)))
            self.assertEqual(fake.scatter_order, expected)
            self.assertEqual(fake.work_order, expected)

    def test_all_loss_paths_pass_the_prior(self):
        gpt = (ROOT / "track_1_short/model/gpt.py").read_text()
        self.assertEqual(gpt.count("self.lm_head.x_s, self.lm_head.w_s, self.lm_head.grad_s, *prior,"), 2)
        self.assertIn("raw = add_prior_(raw.contiguous(), cols, vals, self.bigram_prior_beta)", gpt)
        for path in ("track_1_short/perf/kernels/cross_entropy.py", "track_1_short/perf/kernels/sampled_cross_entropy.py"):
            src = (ROOT / path).read_text()
            self.assertIn("prior_cols=None, prior_vals=None, prior_beta=None", src)
            self.assertIn("grad_beta", src)


if __name__ == "__main__":
    unittest.main()

"""CPU checks of the evaluation sweep (train_gpt_rtx.py eval_sweep, model temperatures); no GPU needed.

The trainer imports CUDA at load, so its sweep settings are read from the source. GPU runs remain the
real test: these check the temperature arithmetic and that the official validation is left alone.
"""
import ast
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from track_1_short.config import TRAINING_STAGES, WS_POST_YARN_EXT  # noqa: E402
from track_1_short.model.layers import cross_entropy_at_temperatures  # noqa: E402


def constants(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name) and t.id.startswith("EVAL_SWEEP")}


class TemperatureTests(unittest.TestCase):
    def test_temperature_one_is_exactly_the_plain_loss(self):
        logits, targets = torch.randn(64, 50304) * 5, torch.randint(0, 50304, (64,))
        swept = cross_entropy_at_temperatures(logits, targets, (1.0,))
        self.assertTrue(torch.equal(swept[0], F.cross_entropy(logits, targets, reduction="none")))

    def test_each_row_is_the_loss_at_its_temperature(self):
        logits, targets = torch.randn(32, 1000, dtype=torch.float64), torch.randint(0, 1000, (32,))
        temperatures = (0.9, 1.0, 1.1)
        swept = cross_entropy_at_temperatures(logits, targets, temperatures)
        self.assertEqual(tuple(swept.shape), (3, 32))
        for row, t in zip(swept, temperatures):
            expected = torch.logsumexp(logits / t, dim=-1) - logits.gather(1, targets[:, None])[:, 0] / t
            torch.testing.assert_close(row, expected)

    def test_masked_tokens_stay_masked(self):
        # The canonical mask sets dropped tokens to -60; at any swept temperature they stay negligible.
        logits = torch.full((1, 4), -60.0, dtype=torch.float64)
        logits[0, :2] = torch.tensor([10.0, 12.0])
        temperatures = (0.9, 1.1)
        for t, loss in zip(temperatures, cross_entropy_at_temperatures(logits, torch.tensor([1]), temperatures)[:, 0]):
            two_way = F.cross_entropy(logits[:, :2] / t, torch.tensor([1]))
            self.assertAlmostEqual(loss.item(), two_way.item(), places=12)


class SweepSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = constants(ROOT / "train_gpt_rtx.py")
        cls.trainer = ast.parse((ROOT / "train_gpt_rtx.py").read_text(encoding="utf-8"))

    def test_first_entry_reproduces_the_official_validation(self):
        self.assertEqual(self.settings["EVAL_SWEEP_WINDOW_OFFSETS"][0], (0, 0))
        self.assertIn(1.00, self.settings["EVAL_SWEEP_TEMPERATURES"])
        # The official final validation: the extension stage's short window, the long window widened.
        self.assertEqual((TRAINING_STAGES[-1].window_sizes[0], WS_POST_YARN_EXT), (6, 20))

    def test_sweep_runs_after_the_official_summary(self):
        main = next(n for n in self.trainer.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        source = ast.unparse(main)
        self.assertLess(source.index("final_val_loss="), source.index("eval_sweep("))
        # The last teardown: the memory preflight exits earlier through its own.
        self.assertLess(source.index("eval_sweep("), source.rindex("dist.destroy_process_group()"))

    def test_official_path_keeps_the_plain_loss(self):
        gpt = ast.parse((ROOT / "track_1_short/model/gpt.py").read_text(encoding="utf-8"))
        init = next(n for n in ast.walk(gpt) if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                    and "vocab_size" in ast.unparse(n))
        self.assertIn("self.eval_temperatures: tuple[float, ...] | None = None", ast.unparse(init))
        loss = next(n for n in ast.walk(gpt) if isinstance(n, ast.FunctionDef) and n.name == "_loss")
        self.assertIn("if self.eval_temperatures is None:\n", ast.unparse(loss))


if __name__ == "__main__":
    unittest.main()

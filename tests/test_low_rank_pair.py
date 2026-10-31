import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_low_rank_pair", ROOT / "tools/run_low_rank_pair.py")
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)


class PairedRunTests(unittest.TestCase):
    def exercise_pair(self, fail_candidate=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            state = {"branch": pair.EXPERIMENT, "runs": [], "checks": []}

            def git(*args):
                if args[0] == "branch":
                    return state["branch"]
                if args[0] == "status":
                    return ""
                if args[0] == "switch":
                    state["branch"] = args[1]
                    return ""
                if args[0] == "merge-base" or args == ("rev-parse", "baseline"):
                    return pair.BASELINE_COMMIT
                if args == ("rev-parse", "HEAD"):
                    return pair.BASELINE_COMMIT if state["branch"] == "baseline" else "candidate-commit"
                raise AssertionError(args)

            def run(command, **kwargs):
                if command[0] != "bash":
                    state["checks"].append(command)
                    return
                self.assertEqual(len(state["checks"]), 2)
                state["runs"].append(state["branch"])
                if fail_candidate and state["branch"] == pair.EXPERIMENT:
                    raise subprocess.CalledProcessError(1, command)
                idx = len(state["runs"])
                (root / "logs" / f"run-{idx}.console.txt").write_text(
                    f"step:1194/1194 val_loss:3.29\nfinal_val_loss={3.3 - idx * .01} training_time_ms={830000 - idx * 5000}\n")

            with patch.object(pair, "ROOT", root), patch.object(pair, "git", git), \
                    patch.object(pair.subprocess, "run", run), patch.object(sys, "argv", ["pair", "--seeds", "999"]):
                if fail_candidate:
                    with self.assertRaises(subprocess.CalledProcessError):
                        pair.main()
                else:
                    pair.main()
            self.assertEqual(state["runs"], ["baseline", pair.EXPERIMENT])
            self.assertEqual(state["branch"], pair.EXPERIMENT)
            reports = list((root / "logs/pairs").glob("*/seed-999.json"))
            self.assertEqual(len(reports), 1)
            return json.loads(reports[0].read_text())

    def test_baseline_then_candidate_with_commits_and_results(self):
        report = self.exercise_pair()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["baseline"]["commit"], pair.BASELINE_COMMIT)
        self.assertEqual(report["candidate"]["commit"], "candidate-commit")
        self.assertEqual(report["seconds_delta"], -5)

    def test_failure_keeps_baseline_result_and_restores_branch(self):
        report = self.exercise_pair(fail_candidate=True)
        self.assertEqual(report["status"], "failed")
        self.assertIn("baseline", report)
        self.assertNotIn("candidate", report)

    def test_stale_short_run_flags_are_cleared(self):
        with patch.dict(os.environ, RTX_MAX_STEPS="1", NUM_SCHEDULED_ITERATIONS="10", RTX_PREFLIGHT="1",
                        RTX_MICROBATCH_TOKENS="2048", RTX_UNKNOWN_EXPERIMENT="1"):
            env = pair.run_env(999)
        for flag in ("RTX_MAX_STEPS", "NUM_SCHEDULED_ITERATIONS", "RTX_PREFLIGHT", "RTX_UNKNOWN_EXPERIMENT"):
            self.assertNotIn(flag, env)
        self.assertEqual(env["RTX_MICROBATCH_TOKENS"], "16384")
        self.assertEqual(env["TRAIN_SEED"], "999")

    def test_short_runs_and_missing_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "console.txt"
            for text in ("crash", "step:2/2 val_loss:8.0\nfinal_val_loss=8.0 training_time_ms=2000"):
                log.write_text(text)
                with self.assertRaises(RuntimeError):
                    pair.read_result(log)
            log.write_text("step:1194/1194 val_loss:3.294980\nfinal_val_loss=3.294980 training_time_ms=825500")
            self.assertEqual(pair.read_result(log), dict(final_val_loss=3.294980, training_seconds=825.5))

    def test_deltas_use_the_new_pair(self):
        result = pair.paired_deltas(dict(final_val_loss=3.295, training_seconds=825),
                                    dict(final_val_loss=3.294, training_seconds=820))
        self.assertAlmostEqual(result["loss_delta"], -.001)
        self.assertEqual(result["seconds_delta"], -5)
        self.assertAlmostEqual(result["time_change_percent"], -100 * 5 / 825)


if __name__ == "__main__":
    unittest.main()

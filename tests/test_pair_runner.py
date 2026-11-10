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
spec = importlib.util.spec_from_file_location("run_pair", ROOT / "tools/run_pair.py")
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)
EXPERIMENT = "experiment/some-idea"


def fake_git(state):
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
            return pair.BASELINE_COMMIT if state["branch"] == "baseline" else state.get("head", "candidate-commit")
        raise AssertionError(args)
    return git


class PairedRunTests(unittest.TestCase):
    def exercise_pair(self, fail_candidate=False, fail_upload=False, argv=("--seeds", "999")):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            state = {"branch": EXPERIMENT, "runs": [], "checks": [], "uploads": []}

            def run(command, **kwargs):
                if command[1] == pair.UPLOADER:
                    # Uploads happen after training, from the candidate checkout's uploader.
                    self.assertEqual(state["branch"], EXPERIMENT)
                    state["uploads"].append(command)
                    if fail_upload:
                        raise subprocess.CalledProcessError(1, command)
                    return
                if command[0] != "bash":
                    self.assertEqual(state["branch"], EXPERIMENT)  # gates run on the candidate
                    state["checks"].append(command[2:])
                    return
                self.assertEqual(state["checks"], [["tests/check_rtx5090.py", "--seed", "999"]])
                self.assertNotIn("RTX_ROUTING", kwargs["env"])
                self.assertEqual(kwargs["env"]["RTX_WANDB_UPLOAD"], "0")  # no self-named baseline upload
                self.assertEqual(state["uploads"], [])
                state["runs"].append(state["branch"])
                if fail_candidate and state["branch"] == EXPERIMENT:
                    raise subprocess.CalledProcessError(1, command)
                idx = len(state["runs"])
                (root / "logs" / f"run-{idx}.console.txt").write_text(
                    "step:1194/1194 val_loss:3.29\nfinal_val_loss="
                    f"{3.3 - idx * .01} training_time_ms={830000 - idx * 5000} peak_memory_allocated=19.99 GiB\n")

            with patch.object(pair, "ROOT", root), patch.object(pair, "git", fake_git(state)), \
                    patch.object(pair.subprocess, "run", run), patch.object(sys, "argv", ["pair", *argv]), \
                    patch.dict(os.environ, RTX_ROUTING="baseline", WANDB_UPLOAD_PYTHON="uploader-python"):
                if fail_candidate:
                    with self.assertRaises(subprocess.CalledProcessError):
                        pair.main()
                else:
                    pair.main()
            self.assertEqual(state["runs"], ["baseline", EXPERIMENT])
            self.assertEqual(state["branch"], EXPERIMENT)
            reports = list((root / "logs/pairs").glob("*/seed-999.json"))
            self.assertEqual(len(reports), 1)
            self.uploads = state["uploads"]
            return json.loads(reports[0].read_text())

    @staticmethod
    def flag(command, name):
        return command[command.index(name) + 1]

    def test_both_arms_upload_under_the_experiment_label(self):
        report = self.exercise_pair()
        names = [self.flag(c, "--name") for c in self.uploads]
        self.assertEqual(names, ["baseline-for-some-idea-1gpu-seed-999", "some-idea-1gpu-seed-999"])
        self.assertEqual([self.flag(c, "--training-commit") for c in self.uploads],
                         [pair.BASELINE_COMMIT, "candidate-commit"])
        self.assertEqual([c[2] for c in self.uploads],
                         [report["baseline"]["console_log"], report["candidate"]["console_log"]])
        for command in self.uploads:
            self.assertEqual(command[0], "uploader-python")
            self.assertEqual((self.flag(command, "--entity"), self.flag(command, "--project"),
                              self.flag(command, "--seed")), ("i-learn", "speedrun", "999"))
        self.assertEqual(report["baseline"]["wandb_name"], "baseline-for-some-idea-1gpu-seed-999")
        self.assertTrue(report["candidate"]["wandb_uploaded"])
        self.assertEqual(report["label"], "some-idea")

    def test_label_override_and_invalid_labels(self):
        self.exercise_pair(argv=("--seeds", "999", "--label", "attnres-zero-init"))
        self.assertEqual([self.flag(c, "--name") for c in self.uploads],
                         ["baseline-for-attnres-zero-init-1gpu-seed-999", "attnres-zero-init-1gpu-seed-999"])
        for bad in ("Baseline", "baseline-x", "a b", "x_y", ".x"):
            with self.subTest(label=bad), patch.object(pair, "git", fake_git({"branch": EXPERIMENT})), \
                    patch.object(pair.subprocess, "run", side_effect=AssertionError("ran")), \
                    patch.object(sys, "argv", ["pair", "--label", bad]):
                with self.assertRaises(RuntimeError):
                    pair.main()

    def test_upload_failure_keeps_complete_pair(self):
        report = self.exercise_pair(fail_upload=True)
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["baseline"]["wandb_uploaded"])
        self.assertFalse(report["candidate"]["wandb_uploaded"])

    def test_baseline_then_candidate_with_commits_and_results(self):
        report = self.exercise_pair()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["experiment"], EXPERIMENT)
        self.assertEqual(report["baseline"]["commit"], pair.BASELINE_COMMIT)
        self.assertEqual(report["candidate"]["commit"], "candidate-commit")
        self.assertEqual(report["seconds_delta"], -5)
        self.assertAlmostEqual(report["loss_delta"], -.01)
        self.assertEqual(report["candidate"]["peak_allocated_gib"], 19.99)

    def test_failure_keeps_baseline_result_and_restores_branch(self):
        report = self.exercise_pair(fail_candidate=True)
        self.assertEqual(report["status"], "failed")
        self.assertIn("baseline", report)
        self.assertNotIn("candidate", report)
        # The finished baseline is still imported under its pair name.
        self.assertEqual([self.flag(c, "--name") for c in self.uploads], ["baseline-for-some-idea-1gpu-seed-999"])

    def test_refuses_baseline_detached_or_baseline_commit(self):
        for branch, head in (("baseline", "candidate-commit"), ("", "candidate-commit"),
                             (EXPERIMENT, pair.BASELINE_COMMIT)):
            with self.subTest(branch=branch, head=head):
                state = {"branch": branch, "head": head}
                with patch.object(pair, "git", fake_git(state)), \
                        patch.object(pair.subprocess, "run", side_effect=AssertionError("ran")), \
                        patch.object(sys, "argv", ["pair"]):
                    with self.assertRaises(RuntimeError):
                        pair.main()

    def test_stale_short_run_flags_are_cleared(self):
        with patch.dict(os.environ, RTX_MAX_STEPS="1", NUM_SCHEDULED_ITERATIONS="10", RTX_PREFLIGHT="1",
                        RTX_MICROBATCH_TOKENS="2048", RTX_ROUTING="static", RTX_UNKNOWN_EXPERIMENT="1"):
            env = pair.run_env(999)
        for flag in ("RTX_MAX_STEPS", "NUM_SCHEDULED_ITERATIONS", "RTX_PREFLIGHT", "RTX_ROUTING",
                     "RTX_UNKNOWN_EXPERIMENT"):
            self.assertNotIn(flag, env)
        self.assertEqual(env["RTX_MICROBATCH_TOKENS"], "16384")
        self.assertEqual(env["RTX_VAL_SEQ_LEN"], "65536")
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

    def test_candidate_env_reaches_only_the_gate_and_candidate(self):
        seen = []
        state = {"branch": EXPERIMENT}

        def run(command, **kwargs):
            if command[0] == "bash" or command[1] != pair.UPLOADER:
                seen.append((state["branch"], command[2] if command[0] != "bash" else "train",
                             kwargs["env"].get("RTX_VOCAB_PRUNE_MASS")))
            if command[0] == "bash":
                raise subprocess.CalledProcessError(1, command)  # stop after the first arm

        with patch.object(pair, "git", fake_git(state)), patch.object(pair.subprocess, "run", run),                 patch.object(sys, "argv", ["pair", "--candidate-env", "RTX_VOCAB_PRUNE_MASS=0.05"]):
            with self.assertRaises(subprocess.CalledProcessError):
                pair.main()
        self.assertEqual(seen, [(EXPERIMENT, "tests/check_rtx5090.py", "0.05"), ("baseline", "train", None)])
        for bad in ("MASS=0.05", "RTX_X"):
            with self.subTest(bad=bad), patch.object(sys, "argv", ["pair", "--candidate-env", bad]):
                with self.assertRaises(SystemExit):
                    pair.main()

    def test_gate_failure_prevents_training(self):
        state = {"branch": EXPERIMENT}

        def run(command, **kwargs):
            raise subprocess.CalledProcessError(1, command)

        with patch.object(pair, "git", fake_git(state)), patch.object(pair.subprocess, "run", side_effect=run) as calls, \
                patch.object(sys, "argv", ["pair", "--seeds", "999"]):
            with self.assertRaises(subprocess.CalledProcessError):
                pair.main()
        self.assertEqual([call.args[0][2] for call in calls.call_args_list], ["tests/check_rtx5090.py"])
        self.assertEqual(state["branch"], EXPERIMENT)


if __name__ == "__main__":
    unittest.main()

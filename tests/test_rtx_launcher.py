"""Exercise launcher control flow with fake training/upload commands; no network/GPU."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASH = ("C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash"))


@unittest.skipUnless(BASH and Path(BASH).exists(), "Bash required")
class LauncherTests(unittest.TestCase):
    def launch(self, **overrides):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "run_rtx.sh").write_bytes((ROOT / "run_rtx.sh").read_bytes())
            hook = root / "hook.sh"
            hook.write_text(
                'git() { echo training-commit; }\n'
                'torchrun() { echo "FAKE TRAINING"; return "${TEST_TRAIN_EXIT:-0}"; }\n'
                'fake_upload() { printf "%s\\n" "$@" > "$TEST_RECORD"; '
                'return "${TEST_UPLOAD_EXIT:-0}"; }\n', encoding="utf-8")
            record = root / "upload.txt"
            env = dict(os.environ, BASH_ENV=hook.as_posix(), TEST_RECORD=record.as_posix(),
                       WANDB_UPLOAD_PYTHON="fake_upload", RTX_WANDB_UPLOAD="1",
                       RTX_PREFLIGHT="0", TRAIN_SEED="999")
            env.update(overrides)
            result = subprocess.run([BASH, "run_rtx.sh"], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=30)
            args = record.read_text() if record.exists() else None
            logs = list((root / "logs").glob("*.console.txt"))
            return result, args, len(logs)

    def test_success_uploads_with_launch_metadata(self):
        result, args, count = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--entity\ni-learn\n--project\nspeedrun\n", args)
        self.assertIn("--seed\n999\n--training-commit\ntraining-commit", args)
        self.assertEqual(count, 1)

    def test_training_failure_does_not_upload(self):
        result, args, _ = self.launch(TEST_TRAIN_EXIT="7")
        self.assertEqual(result.returncode, 7)
        self.assertIsNone(args)

    def test_names_follow_each_seed(self):
        # This branch's runs are the per-head V/O whitening experiment, never the baseline.
        for seed in (999, 1000, 1001, 1002, 1003):
            with self.subTest(seed=seed):
                result, args, _ = self.launch(TRAIN_SEED=str(seed))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"--name\nvo-heads-1gpu-seed-{seed}\n", args)

    def test_missing_seed_is_not_mislabeled(self):
        result, args, _ = self.launch(TRAIN_SEED="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--name\nvo-heads-1gpu-seed-unseeded\n", args)
        self.assertNotIn("--seed\n", args)

    def test_preflight_and_optout_skip_upload(self):
        for values in ({"RTX_PREFLIGHT": "1"}, {"RTX_WANDB_UPLOAD": "0"}):
            with self.subTest(values=values):
                result, args, _ = self.launch(**values)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIsNone(args)

    def test_upload_failure_preserves_training_success(self):
        result, args, _ = self.launch(TEST_UPLOAD_EXIT="4")
        self.assertEqual(result.returncode, 0)
        self.assertIsNotNone(args)
        self.assertIn("training succeeded but W&B upload failed", result.stderr)
        self.assertIn("Retry after login/setup:", result.stderr)


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock


spec = importlib.util.spec_from_file_location("upload_wandb", Path(__file__).resolve().parents[1] / "tools/upload_wandb.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
HEADER = ("rtx-local-v2: one GPU (NVIDIA GeForce RTX 5090), 16384-token microbatches "
          "[8, 16, 24, 20, 8] per step by stage, n-gram table 2,643,840 rows (1/32 of the "
          "record's), validation 65536-token forwards, MLP tile 128x128, eval CE slab 4096 rows, steps 2/1194")
LOG = HEADER + "\n" + "\n".join([
    "step:1/2 train_loss_per_token:17.1446 train_time:568ms step_avg:568.13ms",
    "step:2/2 train_loss_per_token:17.0485 train_time:938ms step_avg:469.23ms",
    "WARNING: background canonical mask build failed (9), building it inline",
    "step:2/2 val_loss:8.883883 train_time:281930ms step_avg:140965.04ms",
    "peak memory allocated: 20489 MiB reserved: 25158 MiB",
    "MEMORY verdict: headroom 2.00 GiB: PASS",
    "final_val_loss=8.883883 training_time_ms=281930 peak_memory_allocated=20.01 GiB",
])


class UploadTests(unittest.TestCase):
    def test_whitening_metadata_comes_from_log(self):
        for mode, interval in (("standard", 1), ("gram_pair", 2)):
            line = f"MLP_WHITENING mode={mode} dtype=bf16 maps=6 gram_refresh_interval={interval}"
            result = module.parse_log(LOG.replace(HEADER, HEADER + "\n" + line))
            self.assertEqual(result["config"]["mlp_whitening"], mode)
            self.assertEqual(result["config"]["gram_refresh_interval"], interval)
        self.assertNotIn("mlp_whitening", module.parse_log(LOG)["config"])

    def test_invalid_whitening_metadata_is_rejected(self):
        good = "MLP_WHITENING mode=gram_pair dtype=bf16 maps=6 gram_refresh_interval=2"
        for line in (good + "\n" + good, good.replace("interval=2", "interval=1"),
                     good.replace("mode=gram_pair", "mode=typo")):
            with self.assertRaises(ValueError):
                module.parse_log(LOG.replace(HEADER, HEADER + "\n" + line))

    def test_merge_same_step_and_preserve_timing(self):
        result = module.parse_log(LOG)
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(result["history"][-1]["train/loss_per_token"], 17.0485)
        self.assertEqual(result["history"][-1]["validation/loss"], 8.883883)
        self.assertEqual(result["summary"]["training_time_ms"], 281930)
        self.assertTrue(result["summary"]["canonical_mask_fallback"])
        self.assertFalse(result["config"]["full_schedule"])
        self.assertNotIn("train_seed", result["config"])

    def test_source_snapshot_ignored(self):
        self.assertEqual(module.parse_log('print("final_val_loss=nan")\n' + LOG), module.parse_log(LOG))

    def test_reject_incomplete_duplicate_and_nonfinite(self):
        for text in (LOG.split("final_val_loss=")[0], LOG + "\n" + LOG,
                     LOG.replace("17.0485", "nan"), LOG.replace("281930ms", "infms"),
                     LOG.replace("step:1/2", "step:3/2"),
                     LOG.replace("final_val_loss=8.883883", "final_val_loss=1.0")):
            with self.subTest(text=text[-60:]), self.assertRaises(ValueError):
                module.parse_log(text)

    def test_full_run_and_no_warning(self):
        result = module.parse_log(LOG.replace("steps 2/1194", "steps 2/2").replace(
            "WARNING: background canonical mask build failed (9), building it inline", ""))
        self.assertTrue(result["config"]["full_schedule"])
        self.assertFalse(result["summary"]["canonical_mask_fallback"])

    def test_sdk_contract(self):
        payload = module.parse_log(LOG)
        payload["config"]["source_sha256"] = "a" * 64
        sdk = MagicMock()
        run = sdk.init.return_value.__enter__.return_value
        run.url = "https://wandb.ai/test/run"
        args = SimpleNamespace(project="test", entity=None, name="pilot")
        self.assertEqual(module.upload(payload, args, sdk), run.url)
        self.assertEqual(run.log.call_count, 2)
        for call in run.log.call_args_list:
            point = call.args[0]
            self.assertEqual(set(point), {"optimizer_step", "training_seconds", "training_minutes",
                                         "train/loss_per_token", "train/loss_vs_seconds", "train/loss_vs_minutes"})
            self.assertEqual(point["train/loss_per_token"], point["train/loss_vs_seconds"])
            self.assertEqual(point["train/loss_per_token"], point["train/loss_vs_minutes"])
            self.assertAlmostEqual(point["training_minutes"], point["training_seconds"] / 60)
        run.define_metric.assert_any_call("optimizer_step", hidden=True, summary="none")
        run.define_metric.assert_any_call("training_seconds", hidden=True, summary="none")
        run.define_metric.assert_any_call("training_minutes", hidden=True, summary="none")
        run.define_metric.assert_any_call("train/loss_per_token", step_metric="optimizer_step")
        run.define_metric.assert_any_call("train/loss_vs_seconds", step_metric="training_seconds")
        run.define_metric.assert_any_call("train/loss_vs_minutes", step_metric="training_minutes")
        self.assertEqual(run.log.call_args_list[-1].args[0]["training_seconds"], 0.938)
        run.summary.update.assert_called_once_with(payload["summary"])
        self.assertEqual(sdk.init.call_args.kwargs["resume"], "never")
        self.assertTrue(sdk.Settings.call_args.kwargs["x_disable_stats"])
        self.assertTrue(sdk.Settings.call_args.kwargs["disable_git"])
        run.log_artifact.assert_not_called()

    def test_train_time_not_overwritten_by_validation(self):
        result = module.parse_log(LOG)
        self.assertEqual(result["history"][-1]["train/elapsed_ms"], 938)
        self.assertEqual(result["summary"]["training_time_ms"], 281930)

    def test_validation_only_rows_are_not_uploaded_as_curves(self):
        payload = module.parse_log(LOG)
        payload["config"]["source_sha256"] = "b" * 64
        del payload["history"][-1]["train/loss_per_token"]
        sdk = MagicMock()
        module.upload(payload, SimpleNamespace(project="test", entity=None, name=None), sdk)
        run = sdk.init.return_value.__enter__.return_value
        self.assertEqual(run.log.call_count, 1)
        run.summary.update.assert_called_once_with(payload["summary"])


if __name__ == "__main__":
    unittest.main()

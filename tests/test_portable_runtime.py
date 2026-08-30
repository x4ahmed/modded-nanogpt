from __future__ import annotations

import contextlib
import io
import random
import unittest

from portable_runtime import (
    FULL_SCHEDULE_STEPS,
    NATIVE_VAL_BATCH_SIZE,
    PORTABLE_VAL_BATCH_SIZE,
    format_final_summary,
    make_run_plan,
    parse_runtime_config,
    seed_everything,
    validation_batch_size,
    validation_sequence_length,
)

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None


class RuntimeParserTests(unittest.TestCase):
    def parse_error(self, argv, env):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_runtime_config(argv, env)

    def test_native_defaults_preserve_randomness(self):
        config = parse_runtime_config([], {})
        self.assertFalse(config.portable)
        self.assertIsNone(config.seed)
        self.assertIsNone(config.max_steps)

    def test_portable_requires_run_id_and_seed(self):
        self.parse_error([], {"PORTABLE": "1"})
        self.parse_error(["--run-id", "run"], {"PORTABLE": "1"})

    def test_python_hash_seed_is_recorded_but_not_enforced(self):
        # Nothing in the training path depends on hash ordering, so a mismatched or
        # absent PYTHONHASHSEED must not fail an otherwise-valid run. It is captured
        # for the manifest instead.
        argv = ["--run-id", "run", "--seed", "7"]
        mismatched = parse_runtime_config(argv, {"PORTABLE": "1", "PYTHONHASHSEED": "8"})
        self.assertEqual(mismatched.seed, 7)
        self.assertEqual(mismatched.python_hash_seed, "8")
        absent = parse_runtime_config(argv, {"PORTABLE": "1"})
        self.assertEqual(absent.seed, 7)
        self.assertIsNone(absent.python_hash_seed)

    def test_portable_accepts_seed_boundaries(self):
        for seed in (0, 2**32 - 1):
            config = parse_runtime_config(
                ["--run-id", "baseline.run-1", "--seed", str(seed)],
                {"PORTABLE": "1", "PYTHONHASHSEED": str(seed)},
            )
            self.assertEqual(config.seed, seed)

    def test_seed_and_step_bounds_are_strict(self):
        base_env = {"PORTABLE": "1", "PYTHONHASHSEED": "0"}
        self.parse_error(["--run-id", "run", "--seed", "-1"], base_env)
        self.parse_error(["--run-id", "run", "--seed", str(2**32)], base_env)
        for value in ("0", str(FULL_SCHEDULE_STEPS + 1)):
            self.parse_error(
                ["--run-id", "run", "--seed", "0", "--max-steps", value],
                base_env,
            )

    def test_run_id_is_path_safe(self):
        env = {"PORTABLE": "1", "PYTHONHASHSEED": "1"}
        for unsafe in ("../run", "a/b", "a\\b", ".", "name with spaces"):
            self.parse_error(["--run-id", unsafe, "--seed", "1"], env)

    def test_native_rejects_silently_ignored_portable_flags(self):
        self.parse_error(["--seed", "1"], {})
        self.parse_error(["--max-steps", "1"], {})

    def test_torchrun_local_rank_argument_is_accepted(self):
        config = parse_runtime_config(["--local-rank=3"], {"LOCAL_RANK": "3"})
        self.assertFalse(config.portable)


class RunPlanTests(unittest.TestCase):
    def test_full_plan_has_1285_updates_and_terminal_validation(self):
        plan = make_run_plan(FULL_SCHEDULE_STEPS, None)
        iterations = list(plan.iterations())
        self.assertEqual(len(iterations), FULL_SCHEDULE_STEPS + 1)
        self.assertEqual(iterations[-1], (FULL_SCHEDULE_STEPS, True))
        self.assertTrue(plan.complete_schedule)

    def test_short_plan_truncates_without_becoming_complete(self):
        plan = make_run_plan(FULL_SCHEDULE_STEPS, 3)
        self.assertEqual(list(plan.iterations()), [(0, False), (1, False), (2, False), (3, True)])
        self.assertFalse(plan.complete_schedule)

    def test_schedule_length_cannot_be_rescaled(self):
        with self.assertRaises(ValueError):
            make_run_plan(10, 3)


class PackingAndSeedTests(unittest.TestCase):
    def test_validation_packing(self):
        self.assertEqual(validation_batch_size(False), NATIVE_VAL_BATCH_SIZE)
        self.assertEqual(validation_batch_size(True), PORTABLE_VAL_BATCH_SIZE)
        for world_size in (1, 2, 4, 8):
            accum = 8 // world_size
            self.assertEqual(
                validation_sequence_length(PORTABLE_VAL_BATCH_SIZE, world_size, accum),
                65_536,
            )
            self.assertEqual(
                validation_sequence_length(NATIVE_VAL_BATCH_SIZE, world_size, accum),
                262_144,
            )

    def test_seed_calls_every_rng_once_with_no_rank_offset(self):
        calls = []

        class Random:
            def seed(self, value):
                calls.append(("random", value))

        class NumpyRandom:
            def seed(self, value):
                calls.append(("numpy", value))

        class Numpy:
            random = NumpyRandom()

        class Cuda:
            def manual_seed_all(self, value):
                calls.append(("cuda", value))

        class Torch:
            cuda = Cuda()

            def manual_seed(self, value):
                calls.append(("torch", value))

        seed_everything(41, random_module=Random(), numpy_module=Numpy(), torch_module=Torch())
        self.assertEqual(
            calls,
            [("random", 41), ("numpy", 41), ("torch", 41), ("cuda", 41)],
        )

    def test_final_summary_has_six_loss_decimals_and_memory_bytes(self):
        summary = format_final_summary(3.2, 1234.5, 10, 20)
        self.assertIn("final_val_loss=3.200000", summary)
        self.assertIn("training_time_ms=1234", summary)
        self.assertIn("peak_memory_allocated_bytes=10", summary)
        self.assertIn("peak_memory_reserved_bytes=20", summary)

    @unittest.skipIf(torch is None or np is None, "NumPy/PyTorch are not installed")
    def test_real_rng_fingerprint_repeats_only_for_same_seed(self):
        def fingerprint(seed):
            seed_everything(
                seed,
                random_module=random,
                numpy_module=np,
                torch_module=torch,
            )
            return (
                random.random(),
                float(np.random.random()),
                tuple(torch.rand(4).tolist()),
            )

        first = fingerprint(123)
        self.assertEqual(first, fingerprint(123))
        self.assertNotEqual(first, fingerprint(124))


if __name__ == "__main__":
    unittest.main()

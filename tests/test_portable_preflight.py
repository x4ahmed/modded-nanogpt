from __future__ import annotations

import math
import importlib.util
import contextlib
import io
import unittest
from unittest import mock
from pathlib import Path

from portable_preflight import GIB, memory_gate_passes, memory_headroom_bytes

try:
    import torch
except ImportError:
    torch = None

_PREFLIGHT_PATH = Path(__file__).with_name("rtx_preflight.py")
_SPEC = importlib.util.spec_from_file_location("rtx_preflight_cli", _PREFLIGHT_PATH)
RTX_PREFLIGHT = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(RTX_PREFLIGHT)


class MemoryGateTests(unittest.TestCase):
    def test_gate_uses_total_minus_peak_reserved(self):
        total = 32 * GIB
        self.assertEqual(memory_headroom_bytes(total, 30 * GIB), 2 * GIB)
        self.assertTrue(memory_gate_passes(total, 30 * GIB, 2.0))
        self.assertFalse(memory_gate_passes(total, 30 * GIB + 1, 2.0))

    def test_headroom_must_be_finite_and_nonnegative(self):
        for invalid in (-1.0, math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                memory_gate_passes(32 * GIB, 1, invalid)


class PreflightCliTypeTests(unittest.TestCase):
    def test_seed_boundaries(self):
        self.assertEqual(RTX_PREFLIGHT.bounded_seed("0"), 0)
        self.assertEqual(RTX_PREFLIGHT.bounded_seed(str(2**32 - 1)), 2**32 - 1)
        for value in ("-1", str(2**32), "nope"):
            with self.assertRaises(Exception):
                RTX_PREFLIGHT.bounded_seed(value)

    def test_headroom_validation(self):
        self.assertEqual(RTX_PREFLIGHT.nonnegative_finite("0"), 0.0)
        self.assertEqual(RTX_PREFLIGHT.nonnegative_finite("2.5"), 2.5)
        for value in ("-1", "inf", "nan", "nope"):
            with self.assertRaises(Exception):
                RTX_PREFLIGHT.nonnegative_finite(value)

    def test_bootstrap_sets_child_hash_seed_and_portable_mode(self):
        env = {
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "2",
            "RANK": "0",
        }
        with (
            mock.patch.object(RTX_PREFLIGHT.sys, "argv", ["rtx_preflight.py", "--seed", "9"]),
            mock.patch.dict(RTX_PREFLIGHT.os.environ, env, clear=True),
            mock.patch.object(RTX_PREFLIGHT.os, "execve") as execve,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            RTX_PREFLIGHT.main()
        child_env = execve.call_args.args[2]
        self.assertEqual(child_env["PORTABLE"], "1")
        self.assertEqual(child_env["RTX_PREFLIGHT"], "1")
        self.assertEqual(child_env["_RTX_DEFER_CE_INIT"], "1")
        self.assertEqual(child_env["PYTHONHASHSEED"], "9")
        self.assertEqual(child_env["RTX_MIN_HEADROOM_GIB"], "2.0")

    def test_bootstrap_rejects_fp8_fallback(self):
        env = {
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "2",
            "DISABLE_FP8": "1",
        }
        with (
            mock.patch.object(RTX_PREFLIGHT.sys, "argv", ["rtx_preflight.py", "--seed", "9"]),
            mock.patch.dict(RTX_PREFLIGHT.os.environ, env, clear=True),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            RTX_PREFLIGHT.main()


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CeReferenceTests(unittest.TestCase):
    def test_duplicate_prefix_weight_is_included(self):
        from portable_preflight import _ce_reference

        logits = torch.zeros(2, 8, dtype=torch.bfloat16)
        targets = torch.tensor([2, 3])
        prefix_targets = torch.tensor([2, -1])
        prefix_weight = torch.tensor([0.25])
        losses, gradients = _ce_reference(
            torch,
            logits,
            targets,
            prefix_targets,
            prefix_weight,
            A=23.0,
            B=5.0,
            C=7.5,
        )
        expected = torch.log(torch.tensor(8.0))
        torch.testing.assert_close(losses[0], 1.25 * expected)
        torch.testing.assert_close(losses[1], expected)
        self.assertEqual(gradients.dtype, torch.float8_e5m2)


if __name__ == "__main__":
    unittest.main()

"""CPU-only tests for diagnostic isolation and failure reporting."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

from diagnose_rtx5090 import execute


class RunnerTests(unittest.TestCase):
    def run_child(self, source, timeout=10):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / "case.txt"
            result = execute([sys.executable, "-u", "-c", source], path, timeout)
            return result, path.read_text(encoding="utf-8")

    def test_success_requires_marker_and_clean_exit(self):
        result, log = self.run_child('print(\'RTX_DIAG_RESULT {"status": "pass"}\')')
        self.assertEqual(result["status"], "pass")
        self.assertIn('"status": "pass"', log)

    def test_crash_does_not_accept_earlier_pass(self):
        result, log = self.run_child('print(\'RTX_DIAG_RESULT {"status": "pass"}\'); raise RuntimeError("CUDA simulation")')
        self.assertEqual(result["status"], "fail")
        self.assertIn("CUDA simulation", log)

    def test_reported_failure_overrides_zero_exit(self):
        result, _ = self.run_child('print(\'RTX_DIAG_RESULT {"status": "fail", "error": "NaN"}\')')
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["result"]["error"], "NaN")

    def test_missing_or_malformed_result_fails(self):
        for output in ("", "RTX_DIAG_RESULT nope", "RTX_DIAG_RESULT []"):
            with self.subTest(output=output):
                result, _ = self.run_child(f"print({output!r})")
                self.assertEqual(result["status"], "fail")

    def test_timeout_stops_child_and_next_case_still_runs(self):
        result, _ = self.run_child("import time; time.sleep(60)", timeout=0.3)
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["status"], "fail")
        next_result, _ = self.run_child('print(\'RTX_DIAG_RESULT {"status": "pass"}\')')
        self.assertEqual(next_result["status"], "pass")


if __name__ == "__main__":
    unittest.main()

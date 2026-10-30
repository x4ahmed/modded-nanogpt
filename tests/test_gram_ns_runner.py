"""A failing diagnostic must emit a failure marker and return a nonzero status."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import check_gram_ns


class DiagnosticStatusTests(unittest.TestCase):
    def run_case(self, error=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = io.StringIO()
            with mock.patch.object(check_gram_ns, "ROOT", root), \
                 mock.patch.object(check_gram_ns.os, "chdir"), \
                 mock.patch.object(check_gram_ns, "run_checks", side_effect=error), \
                 contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                status = check_gram_ns.main([])
            log, = (root / "logs").glob("check-gram-ns-*.txt")
            return status, capture.getvalue(), log.read_text()

    def test_success_marker_requires_completed_checks(self):
        status, console, log = self.run_case()
        self.assertEqual(status, 0)
        self.assertIn("GRAM_NS_CHECK status=pass", console)
        self.assertIn("GRAM_NS_CHECK status=pass", log)

    def test_error_and_interrupt_fail(self):
        for error in (RuntimeError("kernel failure"), MemoryError("allocation failure"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                status, console, log = self.run_case(error)
                self.assertEqual(status, 1)
                self.assertIn("GRAM_NS_CHECK status=fail", console)
                self.assertIn("GRAM_NS_CHECK status=fail", log)
                self.assertNotIn("status=pass", log)

    def test_cuda_oom_fails(self):
        import torch
        status, _, log = self.run_case(torch.OutOfMemoryError("synthetic OOM"))
        self.assertEqual(status, 1)
        self.assertIn("synthetic OOM", log)
        self.assertIn("GRAM_NS_CHECK status=fail", log)


if __name__ == "__main__":
    unittest.main()

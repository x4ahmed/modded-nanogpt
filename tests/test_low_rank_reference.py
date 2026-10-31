"""Validate the independent arithmetic oracle and the scalar parity gates."""
import contextlib
import io
import unittest

import torch

from low_rank_reference import ParityReport, compare, fp8_reference, scalar_diagnostic


class LowRankReferenceTests(unittest.TestCase):
    def test_fp8_relu_square_avoids_extra_bf16_rounding(self):
        # Exact FP8 operands give pre = 1 + .125 * .25 = 1.03125.
        # FP32 square 1.0634765625 rounds to FP8 1.125. Rounding it
        # through BF16 first gives 1.0625, an FP8 tie that rounds to 1.0.
        x = torch.tensor([[[1., .125], [-1., -.125]]], dtype=torch.bfloat16)
        up = torch.tensor([[1., .25]], dtype=torch.bfloat16)
        a = torch.ones(1, 1, dtype=torch.bfloat16)
        b = torch.ones(2, 1, dtype=torch.bfloat16)
        one = torch.tensor(1.)
        got, _ = fp8_reference(
            x, up, a, b, one.bfloat16(), torch.zeros_like(x),
            x_scale=one, grad_scale=one, up_scale=one, a_scale=one,
            b_scale=one, post_scale=one, dpre_scale=one)
        expected = torch.tensor([[[1.125, 1.125], [0., 0.]]], dtype=torch.bfloat16)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_exactly_representable_chain_rule_including_lambda(self):
        for p_value in (1., 2.):
            for zero_a in (False, True):
                with self.subTest(p=p_value, zero_a=zero_a):
                    x = torch.full((1, 16, 4), .25, dtype=torch.bfloat16, requires_grad=True)
                    up = torch.full((8, 4), .25, dtype=torch.bfloat16, requires_grad=True)
                    a = torch.full((1, 8, 2), 0. if zero_a else .25, dtype=torch.bfloat16, requires_grad=True)
                    b = torch.full((1, 4, 2), .25, dtype=torch.bfloat16, requires_grad=True)
                    p = torch.tensor(p_value, dtype=torch.bfloat16, requires_grad=True)
                    g = torch.full_like(x, .25)
                    expected = ((x @ up.T).relu().square() @ a[0] * p) @ b[0].T
                    grads = torch.autograd.grad(expected, (x, up, a, b, p), g)
                    one = torch.tensor(1.)
                    got, got_grads = fp8_reference(
                        x, up, a[0], b[0], p, g, x_scale=one, grad_scale=one, up_scale=one,
                        a_scale=one, b_scale=one, post_scale=one, dpre_scale=one)
                    torch.testing.assert_close(got, expected, rtol=0, atol=1e-7)
                    for actual, reference in zip(got_grads, grads):
                        torch.testing.assert_close(actual, reference, rtol=0, atol=1e-7)

    def test_stable_probe_rejects_zero_wrong_sign_and_wrong_gain(self):
        reference = torch.tensor(100.)
        with contextlib.redirect_stdout(io.StringIO()):
            compare("aligned", torch.tensor(101.), reference, .15)
            for broken in (0., -100., 50., 150., float("nan"), float("inf")):
                with self.subTest(broken=broken), self.assertRaises(AssertionError):
                    compare("aligned", torch.tensor(broken), reference, .15)

    def test_random_scalar_cancellation_is_reported_not_hidden(self):
        expected = torch.tensor([1., -.99])
        perturbed = torch.tensor([1.01, -.99])
        # <ones, y> is ill-conditioned, despite only ~0.7% vector error.
        random_expected, random_actual = expected.sum(), perturbed.sum()
        self.assertGreater(((random_actual - random_expected) / random_expected).abs().item(), .9)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            scalar_diagnostic("random", random_actual, random_expected)
            compare("output", perturbed, expected, .10)
            compare("aligned", (perturbed * expected).sum(), expected.square().sum(), .15)
        self.assertIn("gating=false", stream.getvalue())
        self.assertIn("reference=", stream.getvalue())
        self.assertIn("abs_error=", stream.getvalue())

    def test_quantized_reference_check_rejects_large_gradient_error(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(AssertionError):
            compare("quantized-reference/a", torch.tensor([1., 2.5]), torch.tensor([1., 2.]), .03)

    def test_report_prints_later_checks_but_still_fails(self):
        report = ParityReport()
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            report.compare("output", torch.tensor(2.), torch.tensor(1.), .03)
            report.compare("later-gradient", torch.tensor(1.), torch.tensor(1.), .03)
            report.compare("nonfinite", torch.tensor(float("nan")), torch.tensor(1.), .03)
            with self.assertRaises(AssertionError):
                report.finish()
        self.assertEqual(report.failures, ["output", "nonfinite"])
        self.assertIn("PARITY later-gradient", stream.getvalue())
        self.assertIn("LOW_RANK_CHECK status=fail failed_checks=2", stream.getvalue())
        self.assertNotIn("status=pass", stream.getvalue())

    def test_report_accepts_all_passes(self):
        report = ParityReport()
        with contextlib.redirect_stdout(io.StringIO()):
            report.compare("exact", torch.tensor(1.), torch.tensor(1.), 0.)
            report.finish()
        self.assertFalse(report.failures)


if __name__ == "__main__":
    unittest.main()

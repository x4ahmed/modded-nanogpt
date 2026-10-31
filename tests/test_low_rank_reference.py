"""Validate the independent arithmetic oracle and the scalar parity gates."""
import contextlib
import io
import unittest

import torch

from low_rank_reference import compare, fp8_reference, scalar_diagnostic


class LowRankReferenceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

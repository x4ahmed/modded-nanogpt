"""Independent, untimed PyTorch reference for the low-rank FP8 arithmetic.

No custom autograd functions or Triton kernels. Dequantize FP8 operands into
FP32 matmuls and explicitly reproduce BF16 stores / FP8 rounding boundaries.
This is a test oracle, never a production fallback or a timing measurement.
"""
import torch


def quantize(value, scale, dtype):
    limit = torch.finfo(dtype).max
    return (value.float() / scale).clamp(-limit, limit).to(dtype)


def mm(left, right, left_scale, right_scale):
    return ((left.float() @ right.float()) * (left_scale * right_scale)).bfloat16()


@torch.no_grad()
def fp8_reference(x, up, a, b, p, grad, *, x_scale, grad_scale, up_scale,
                  a_scale, b_scale, post_scale, dpre_scale):
    """Return output and VJP for x/up/A/B/p using an independent chain rule."""
    e4, e5 = torch.float8_e4m3fn, torch.float8_e5m2
    xq = quantize(x.flatten(0, 1), x_scale, e4)
    uq, aq, bq = (quantize(w, s, e4) for w, s in ((up, up_scale), (a, a_scale), (b, b_scale)))
    pre = mm(xq, uq.T, x_scale, up_scale)
    # Triton 3.4-3.6 tl.maximum promotes BF16 to FP32. The fused kernel
    # squares in FP32 and goes straight to FP8 (no BF16 post store).
    # A BF16 square here introduces double rounding at FP8 midpoints.
    # https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/language/core.py#L2337-L2342
    post = pre.float().relu().square()
    pq = quantize(post, post_scale, e4)
    folded_a_scale = a_scale * p.float()
    middle = mm(pq, aq, post_scale, folded_a_scale)
    middle_scale = (middle.float().abs().amax() / 448).clamp_min(1e-12)
    mq = quantize(middle, middle_scale, e4)
    out = mm(mq, bq.T, middle_scale, b_scale).view_as(x)

    gq = quantize(grad.flatten(0, 1), grad_scale, e5)
    db = mm(gq.T, mq, grad_scale, middle_scale)
    dm = mm(gq, bq, grad_scale, b_scale)
    dmq = quantize(dm, grad_scale, e5)
    da_unfolded = mm(pq.T, dmq, post_scale, grad_scale)
    da = da_unfolded * p
    dp = (da_unfolded.float() * a.float()).sum().to(p.dtype)
    # The fused dpre epilogue rounds only AFTER multiplying by the ReLU^2
    # derivative. A BF16 dpost temporary here would add a rounding absent in CUDA.
    dpost = (dmq.float() @ aq.float().T) * (grad_scale * folded_a_scale)
    dpre = (2 * dpost * (pq.float() * post_scale).sqrt()).bfloat16()
    dq = quantize(dpre, dpre_scale, e5)
    du = mm(dq.T, xq, dpre_scale, x_scale)
    dx = mm(dq, uq, dpre_scale, up_scale).view_as(x)
    return out, (dx, du, da.unsqueeze(0), db.unsqueeze(0), dp)


def parity_metrics(actual, expected):
    actual, expected = actual.detach().float(), expected.detach().float()
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("nonfinite values or mismatched shapes in parity check")
    error = (actual - expected).norm().item()
    reference = expected.norm().item()
    return error, reference, error / max(reference, 1e-12)


def compare(label, actual, expected, limit):
    error, reference, relative = parity_metrics(actual, expected)
    print(f"PARITY {label} abs_l2={error:.6g} ref_l2={reference:.6g} rel_l2={relative:.6g} limit={limit}", flush=True)
    if relative > limit:
        raise AssertionError(f"{label} exceeds parity tolerance")


class ParityReport:
    """Print all numerical failures before failing the training gate."""
    def __init__(self):
        self.failures = []

    def compare(self, label, actual, expected, limit):
        try:
            compare(label, actual, expected, limit)
        except AssertionError as exc:
            self.failures.append(label)
            print(f"PARITY_FAIL {label}: {exc}", flush=True)

    def finish(self):
        if self.failures:
            print(f"LOW_RANK_CHECK status=fail failed_checks={len(self.failures)}", flush=True)
            raise AssertionError("Failed parity checks: " + ", ".join(self.failures))


def scalar_diagnostic(label, actual, expected):
    """Report the signed random-probe scalar; stable scalar probes gate separately."""
    error, _, relative = parity_metrics(actual, expected)
    print(f"SCALAR_DIAGNOSTIC {label} actual={actual.item():.9g} reference={expected.item():.9g} "
          f"abs_error={error:.6g} relative_error={relative:.6g} gating=false", flush=True)

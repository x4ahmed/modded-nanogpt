"""Focused CUDA check of the production low-rank MLP, before the full smoke test.

Run on a 5090 or H100: python -u tests/check_low_rank_mlp.py
Checks eager and fullgraph-compiled FP8 forwards/backwards against a BF16
reference, including the folded post-lambda and zero-output initialization.
This is a numerical test, not a training-speed measurement.
"""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=16384)
    args = parser.parse_args()
    import torch
    from track_1_short.model.low_rank import LowRankParallelDown
    import triton
    from track_1_short.perf.kernels.mlp import linear_relu_square, prime_stage_cache, quantize_dual_layout_fused
    from low_rank_reference import ParityReport, fp8_reference, scalar_diagnostic

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("This check needs SM90+ (H100 or RTX 5090), not a CPU/older GPU.")
    if args.tokens <= 0 or args.tokens % 128:
        raise ValueError("--tokens must be a positive multiple of 128")
    torch.manual_seed(999)
    print(f"LOW_RANK_CHECK device={torch.cuda.get_device_name()} tokens={args.tokens} "
          f"torch={torch.__version__} triton={triton.__version__}", flush=True)
    # Import the actual scale constants without requiring Hopper's attention
    # package in this standalone MLP check. No attention is executed here.
    os.environ.setdefault("PORTABLE", "1")
    from track_1_short.model.gpt import FP8_MLP_X_SCALE, FP8_GRAD_SCALE
    # The independent oracle uses genuine FP32 matmuls, not TF32 approximations.
    torch.backends.cuda.matmul.allow_tf32 = False
    prime_stage_cache()
    report = ParityReport()
    compare = report.compare

    # Exact operands expose the FP32-square -> FP8 rounding boundary directly
    # in the real fused kernel, independently of the random full-chain checks.
    with torch.no_grad():
        probe_x = torch.zeros(128, 768, device="cuda", dtype=torch.bfloat16)
        probe_up = torch.zeros(2816, 768, device="cuda", dtype=torch.bfloat16)
        probe_x[:, 0], probe_x[:, 1] = 1., .125
        probe_up[:, 0], probe_up[:, 1] = 1., .25
        one = torch.tensor(1., device="cuda")
        pre, _, post, post_t = linear_relu_square(
            probe_x, probe_up, a_f8=probe_x.to(torch.float8_e4m3fn),
            b_f8=probe_up.to(torch.float8_e4m3fn), dequant_scale_ptr=one,
            emit_f8=True, emit_t=True, store_pre=True, store_post_bf=False,
            post_scale=one, post_amax=torch.zeros(1, device="cuda"))
        compare("epilogue/pre", pre, torch.full_like(pre, 1.03125), 0.)
        compare("epilogue/fp32-square-to-fp8", post.float(), torch.full_like(pre, 1.125), 0.)
        compare("epilogue/transpose", post_t.float(), post.float().T, 0.)
        del probe_x, probe_up, pre, post, post_t, one

    part = LowRankParallelDown(2816, 768, device="cuda")
    part.a.data = part.a.data.bfloat16()
    part.b.data = part.b.data.bfloat16()
    x = torch.randn(1, args.tokens, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    up = (torch.randn(2816, 768, device="cuda", dtype=torch.bfloat16) * (0.5 / 768 ** .5)).requires_grad_()
    p = torch.tensor(1.125, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn_like(x) * 0.001
    x_s = torch.tensor(FP8_MLP_X_SCALE, device="cuda")
    g_s = torch.tensor(FP8_GRAD_SCALE, device="cuda")
    up_s = (up.detach().float().abs().amax() / 448).clamp_min(1e-12)
    up_row = (up.detach().float() / up_s).clamp(-448, 448).to(torch.float8_e4m3fn)
    up_col = up_row.T.contiguous().T
    # Same bootstrap activation/gradient scales as GPT.init_mlp_fp8().
    post_s = torch.tensor(1., device="cuda")
    post_amax = torch.zeros(1, device="cuda")
    dpre_s = torch.tensor(FP8_GRAD_SCALE, device="cuda")
    dpre_amax = torch.zeros(1, device="cuda")

    def forward(x, up, p):
        # Match GPT.forward: these are cached operands, not a second autograd
        # path. The custom MLP backward returns dx through the original x below.
        x_row, x_t = quantize_dual_layout_fused(x.detach().flatten(0, 1), x_s, torch.float8_e4m3fn)
        # The production method substitutes all A-specific entries (3,4,7,8).
        fp8_args = (up_row, x_s * up_s, x_row, None, None, post_s, post_amax,
                    None, None, x_t, x_s, up_col, up_s, g_s, dpre_s, dpre_amax, p)
        return part.mlp(x, up, fp8_args)

    params = (x, up, part.a, part.b, p)
    # Eager and compiled must see identical data and weights.
    nonzero_a = torch.empty_like(part.a).normal_(std=.03)
    compiled = torch.compile(forward, fullgraph=True, dynamic=False)
    for name, fn in (("eager", forward), ("compiled", compiled)):
        with torch.no_grad():
            part.a.zero_()
        part.refresh(exact=True, post_lambda=p.detach().float(), grad_scale=g_s)
        out = fn(x, up, p)
        grads = torch.autograd.grad(out, params, g)
        if out.count_nonzero().item() or grads[3].count_nonzero().item():
            raise AssertionError("initial output and B gradient must be zero")
        if not torch.isfinite(grads[2]).all() or not grads[2].count_nonzero().item():
            raise AssertionError("A must receive a finite nonzero initial gradient")
        with torch.no_grad():
            part.a.copy_(nonzero_a)
        part.refresh(exact=True, post_lambda=p.detach().float(), grad_scale=g_s)
        for repeat in range(2):
            out = fn(x, up, p)
            grads = torch.autograd.grad(out, params, g)
            ref = part.reference(x, up) * p
            ref_grads = torch.autograd.grad(ref, params, g)
            # Separate actual kernel/backward errors from expected FP8-vs-BF16
            # quantization differences. Includes the random-probe scalar.
            quant_ref, quant_grads = fp8_reference(
                x, up, part.a[0], part.b[0], p, g,
                x_scale=x_s, grad_scale=g_s, up_scale=up_s,
                a_scale=part.a_scale[0], b_scale=part.b_scale[0],
                post_scale=post_s, dpre_scale=dpre_s)
            compare(f"{name}/{repeat}/quantized-reference/output", out, quant_ref, .03)
            for param_name, got, expected in zip(("input", "up", "a", "b", "post_lambda"), grads, quant_grads):
                compare(f"{name}/{repeat}/quantized-reference/{param_name}", got, expected, .03)
            compare(f"{name}/{repeat}/output", out, ref, .10)
            for param_name, got, expected in zip(("input", "up", "a", "b"), grads[:4], ref_grads[:4]):
                compare(f"{name}/{repeat}/{param_name}", got, expected, .15)
            # A random signed dot product may nearly cancel. Its relative scalar
            # error is diagnostic, not a reliable BF16-accuracy gate by itself.
            scalar_diagnostic(f"{name}/{repeat}/post_lambda/random", grads[-1], ref_grads[-1])
            # With a reference-aligned cotangent, dp is a positive energy sum.
            # Keep the original 15% BF16 tolerance on this well-conditioned probe.
            aligned = ref.detach() * .001
            aligned_out = fn(x, up, p)
            aligned_dp = torch.autograd.grad(aligned_out, params, aligned)[-1]
            aligned_ref = part.reference(x, up) * p
            expected_dp = torch.autograd.grad(aligned_ref, p, aligned)[0]
            if expected_dp.item() <= 0:
                raise AssertionError("The aligned scalar probe must have positive nonzero energy")
            compare(f"{name}/{repeat}/post_lambda/aligned", aligned_dp, expected_dp, .15)
        with torch.no_grad():
            compare(f"{name}/bf16-eval", part.mlp(x, up), part.reference(x, up), .02)
        torch.cuda.synchronize()
    report.finish()
    print("LOW_RANK_CHECK status=pass rank=384 eager=true compiled=true", flush=True)


if __name__ == "__main__":
    main()

"""Device check: every fp8 op a training forward runs, against an eager reference, on this card.

Why this exists. On rtx-local-v2 a bf16 validation forward on fresh weights gives ln(50304) = 10.83,
and every training forward is nan -- immediately, at step 0, before any weight has moved. The only
difference between the two is the fp8 path. `scaled_mm_fast_accum.py` already cleared
use_fast_accum and bf16-output _scaled_mm at the record's shapes, which leaves:

  1. _scaled_mm with an e4m3 OUT dtype -- the CE logit GEMM (cross_entropy.py _ce_logit_gemm). Both
     CE kernels call it, so swapping sampled for full-vocab never excluded it. e4m3 overflow is NaN
     rather than inf, and _scaled_mm ignores scale_result for an fp8 out dtype, so saturation here
     is the implementation's business, not the caller's.
  2. linear_relu_square's fp8 forward (perf/kernels/mlp.py) -- the fused Triton up projection that
     emits the e4m3 relu(pre)^2 in two layouts through TMA stores. Its tiles are 128x128 here
     against the record's 128x256.

Each check recomputes the same arithmetic eagerly from the same fp8 bytes. A few 1e-3 of relative
error is fp8 rounding. NaN is the bug.

Run (one GPU, no torchrun, no PORTABLE):  python tests/rtx/fp8_forward_ops.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root, however this is invoked

import torch

from track_1_short.perf.kernels.mlp import linear_relu_square, mlp_block_n, prime_stage_cache

E4M3_MAX = 448.0
DEV = torch.device("cuda")


def quantize(t):
    scale = (t.abs().amax() / E4M3_MAX).clamp(min=1e-12).float()
    return (t / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), scale


def report(label, got, ref):
    """got / ref as fp32; prints non-finite counts and the worst relative error."""
    bad = int((~torch.isfinite(got)).sum())
    if bad:
        print(f"    {label:22s} NAN/INF in {bad}/{got.numel()} elements")
        return False
    denom = ref.abs().amax().clamp(min=1e-30)
    print(f"    {label:22s} max rel err {((got - ref).abs().amax() / denom).item():.3e}")
    return True


def check_fp8_out_gemm(M, K, N):
    """_ce_logit_gemm: e4m3 x e4m3 -> e4m3, the scales descaling the accumulator."""
    gen = torch.Generator(device=DEV).manual_seed(0)
    x = torch.randn(M, K, generator=gen, device=DEV) * 0.3
    w = torch.randn(N, K, generator=gen, device=DEV) * 0.03
    x_f8, x_s = quantize(x)
    w_f8, w_s = quantize(w)
    w_col = w_f8.T.contiguous().T  # (K, N) column-major, as quantize_lm_head_dual's cache is
    print(f"  CE logit GEMM, e4m3 out  [{M} x {K}] @ [{K} x {N}]")
    out = torch._scaled_mm(x_f8, w_col, out_dtype=torch.float8_e4m3fn,
                           scale_a=x_s, scale_b=w_s, use_fast_accum=True)
    ref = (x_f8.float() * x_s) @ (w_col.float() * w_s)
    report("e4m3 out", out.float(), ref)
    # Same GEMM asked for bf16, as the control: identical inputs, only the out dtype differs.
    out_bf16 = torch._scaled_mm(x_f8, w_col, out_dtype=torch.bfloat16,
                                scale_a=x_s, scale_b=w_s, use_fast_accum=True)
    report("bf16 out (control)", out_bf16.float(), ref)


def check_fused_mlp(M, K, N):
    """linear_relu_square's fp8 forward: the e4m3 relu(pre)^2 in both layouts."""
    gen = torch.Generator(device=DEV).manual_seed(0)
    x = (torch.randn(M, K, generator=gen, device=DEV) * 0.3).bfloat16()
    w1 = (torch.randn(N, K, generator=gen, device=DEV) * 0.03125).bfloat16()
    x_f8, x_s = quantize(x.float())
    w1_f8, w1_s = quantize(w1.float())
    dequant = (x_s * w1_s).reshape(1).clone()
    post_scale = torch.ones(1, dtype=torch.float32, device=DEV)  # the value at the first forward
    post_amax = torch.zeros(1, dtype=torch.float32, device=DEV)
    print(f"  fused MLP up projection  [{M} x {K}] @ [{K} x {N}]  (tiles 128 x "
          f"{mlp_block_n()})")
    _, _, post_f8, post_t = linear_relu_square(
        x, w1, a_f8=x_f8, b_f8=w1_f8, emit_f8=True, emit_t=True,
        dequant_scale_ptr=dequant, post_scale=post_scale, post_amax=post_amax,
        store_pre=False, store_post_bf=False)
    # The kernel's arithmetic: fp32 accumulate over raw fp8 bytes, scale, round to bf16, relu, square,
    # then back to fp32 and quantize with the delayed post scale.
    pre = ((x_f8.float() @ w1_f8.float().T) * dequant).bfloat16()
    post = pre.clamp(min=0)
    post = (post * post).float()
    ref = (post / post_scale).clamp(max=E4M3_MAX)
    report("post, row-major", post_f8.float(), ref)
    report("post, transposed", post_t.float().T, ref)
    print(f"    recorded post amax     {post_amax[0].item():.6g}  (eager {post.amax().item():.6g})")


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    major, minor = torch.cuda.get_device_capability()
    print(f"device  {torch.cuda.get_device_name()}  sm_{major}{minor}")
    print(f"torch   {torch.__version__}")
    prime_stage_cache()
    check_fp8_out_gemm(16384, 768, 10240)
    check_fp8_out_gemm(16384, 768, 50304)
    check_fused_mlp(16384, 768, 2816)
    check_fused_mlp(2048, 768, 2816)


if __name__ == "__main__":
    main()

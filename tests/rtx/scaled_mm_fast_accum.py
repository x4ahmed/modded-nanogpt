"""Device check: does torch._scaled_mm(use_fast_accum=True) give the right answer on this GPU?

Why this exists. Every training forward of the port runs at least one fp8 GEMM with
use_fast_accum=True -- the MLP down projection (perf/kernels/mlp.py `dp_f8`) and the CE logit GEMM
(perf/kernels/cross_entropy.py `_ce_logit_gemm`) -- and the validation forward runs none at all.
That is exactly the split the rtx-local-v2 NaN follows: a bf16 eval forward on fresh weights gives
ln(50304), while every training forward is nan, with fp8 attention on or off and with either CE
kernel. So this checks the one flag directly, on this card, at the shapes the record uses, against
an eager dequantized reference.

fast accum lets the tensor cores accumulate fp8 products at reduced precision inside the MMA
pipeline. It is a throughput/accuracy trade on Hopper; whether SM120 implements it at all is the
question. A few percent of relative error is the trade working as intended. NaN, or an error far
from the slow path's, is not.

Run (one GPU, no torchrun, no PORTABLE):  python tests/rtx/scaled_mm_fast_accum.py
"""
import torch

E4M3_MAX = 448.0


def quantize(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(e4m3 tensor, 0-dim fp32 scale), the per-tensor scaling the record uses."""
    scale = (t.abs().amax() / E4M3_MAX).clamp(min=1e-12).float()
    return (t / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), scale


def check(label: str, M: int, K: int, N: int):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    a = torch.randn(M, K, generator=gen, device=dev, dtype=torch.float32) * 0.3
    # B is column-major (K, N), the layout GPT.init_mlp_fp8's col_major() cache hands _scaled_mm.
    b_store = torch.randn(N, K, generator=gen, device=dev, dtype=torch.float32) * 0.03
    a_f8, a_s = quantize(a)
    b_f8_row, b_s = quantize(b_store)
    b_f8 = b_f8_row.T  # (K, N), strides (1, N)

    # Eager reference over the same fp8 bytes: only the accumulation differs.
    ref = (a_f8.float() * a_s) @ (b_f8.float() * b_s)

    out = {}
    for fast in (False, True):
        try:
            y = torch._scaled_mm(a_f8, b_f8, out_dtype=torch.bfloat16,
                                 scale_a=a_s, scale_b=b_s, use_fast_accum=fast)
        except Exception as exc:  # an unsupported combination raises rather than returning garbage
            out[fast] = f"raised {type(exc).__name__}: {exc}"
            continue
        y32 = y.float()
        nonfinite = int((~torch.isfinite(y32)).sum())
        denom = ref.abs().amax().clamp(min=1e-30)
        rel = ((y32 - ref).abs().amax() / denom).item()
        out[fast] = (f"{'nan/inf in ' + str(nonfinite) + '/' + str(y32.numel()) + ' elements'}"
                     if nonfinite else f"max rel err {rel:.3e}")
    print(f"  {label:28s} [{M} x {K}] @ [{K} x {N}]")
    print(f"    use_fast_accum=False : {out[False]}")
    print(f"    use_fast_accum=True  : {out[True]}")


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    major, minor = torch.cuda.get_device_capability()
    print(f"device  {torch.cuda.get_device_name()}  sm_{major}{minor}")
    print(f"torch   {torch.__version__}")
    print("_scaled_mm, e4m3 x e4m3 -> bf16, per-tensor scales:")
    # The record's own shapes, then a small one in case the fault is shape dependent.
    check("MLP down projection", 16384, 2816, 768)
    check("CE logits, sampled P", 16384, 768, 10240)
    check("CE logits, full vocab", 16384, 768, 50304)
    check("small", 256, 128, 128)


if __name__ == "__main__":
    main()

"""Focused CUDA check of the production low-rank MLP, before the full smoke test.

Run on a 5090 or H100: python -u tests/check_low_rank_mlp.py
Checks eager and fullgraph-compiled FP8 forwards/backwards against a BF16
reference, including the folded post-lambda and zero-output initialization.
This is a numerical test, not a training-speed measurement.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=16384)
    args = parser.parse_args()
    import torch
    from track_1_short.model.low_rank import LowRankParallelDown
    from track_1_short.perf.kernels.mlp import prime_stage_cache, quantize_dual_layout_fused

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("This check needs SM90+ (H100 or RTX 5090), not a CPU/older GPU.")
    if args.tokens <= 0 or args.tokens % 128:
        raise ValueError("--tokens must be a positive multiple of 128")
    torch.manual_seed(999)
    print(f"LOW_RANK_CHECK device={torch.cuda.get_device_name()} tokens={args.tokens}", flush=True)
    prime_stage_cache()
    part = LowRankParallelDown(2816, 768, device="cuda")
    part.a.data = part.a.data.bfloat16()
    part.b.data = part.b.data.bfloat16()
    x = torch.randn(1, args.tokens, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    up = (torch.randn(2816, 768, device="cuda", dtype=torch.bfloat16) * (0.5 / 768 ** .5)).requires_grad_()
    p = torch.tensor(1.125, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn_like(x) * 0.001
    x_s = torch.tensor(8 / 448, device="cuda")
    g_s = torch.tensor((.75 / 8) / 448, device="cuda")
    up_s = (up.detach().float().abs().amax() / 448).clamp_min(1e-12)
    up_row = (up.detach().float() / up_s).clamp(-448, 448).to(torch.float8_e4m3fn)
    up_col = up_row.T.contiguous().T
    post_s = torch.tensor(.04, device="cuda")
    post_amax = torch.zeros(1, device="cuda")
    dpre_s = torch.tensor(1e-7, device="cuda")
    dpre_amax = torch.zeros(1, device="cuda")

    def forward(x, up, p):
        # Match GPT.forward: these are cached operands, not a second autograd
        # path. The custom MLP backward returns dx through the original x below.
        x_row, x_t = quantize_dual_layout_fused(x.detach().flatten(0, 1), x_s, torch.float8_e4m3fn)
        # The production method substitutes all A-specific entries (3,4,7,8).
        fp8_args = (up_row, x_s * up_s, x_row, None, None, post_s, post_amax,
                    None, None, x_t, x_s, up_col, up_s, g_s, dpre_s, dpre_amax, p)
        return part.mlp(x, up, fp8_args)

    def compare(label, actual, expected, limit):
        actual, expected = actual.float(), expected.float()
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError(f"nonfinite {label}")
        relative = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
        print(f"PARITY {label} rel_l2={relative.item():.6g} limit={limit}", flush=True)
        if relative > limit:
            raise AssertionError(f"{label} exceeds FP8 parity tolerance")

    params = (x, up, part.a, part.b, p)
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
            part.a.normal_(std=.03)
        part.refresh(exact=True, post_lambda=p.detach().float(), grad_scale=g_s)
        for repeat in range(2):
            out = fn(x, up, p)
            grads = torch.autograd.grad(out, params, g)
            ref = part.reference(x, up) * p
            ref_grads = torch.autograd.grad(ref, params, g)
            compare(f"{name}/{repeat}/output", out, ref, .10)
            for param_name, got, expected in zip(("input", "up", "a", "b", "post_lambda"), grads, ref_grads):
                compare(f"{name}/{repeat}/{param_name}", got, expected, .15)
        with torch.no_grad():
            compare(f"{name}/bf16-eval", part.mlp(x, up), part.reference(x, up), .02)
        torch.cuda.synchronize()
    print("LOW_RANK_CHECK status=pass rank=384 eager=true compiled=true", flush=True)


if __name__ == "__main__":
    main()

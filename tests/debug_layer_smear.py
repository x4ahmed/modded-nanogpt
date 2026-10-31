"""Locate the layer smear's non-finite gradient (one RTX 5090, a few minutes, no training run).

    python -u tests/debug_layer_smear.py 2>&1 | grep -E "SMEAR_DEBUG|RTX5090_CHECK|Error"

Runs the smoke test (tests/check_rtx5090.py) with token_shift replaced by an opaque op whose backward
forms the weight gradient in fp32 and, for the first backward pass, prints one line per layer: how many
values of the incoming gradient g and of the residual x are non-finite, the largest |g| and where, the
residual's size there and one position earlier, the smallest residual and the largest |g| at it.

Reading it: non-finite g means the gradient reaching the residual is already bad, which the baseline's
own gradients do not reveal; a huge |g| at a near-zero residual means only the shifted product
overflows. If the smoke test passes instead, the compiled token_shift was at fault, not the values.
"""
import argparse
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
LAYERS = 11
state = {"backward_calls": 0}


def report(g: torch.Tensor, x: torch.Tensor, grad_w: torch.Tensor):
    call = state["backward_calls"]
    state["backward_calls"] += 1
    if call >= LAYERS:  # the first backward pass only
        return
    layer = LAYERS - 1 - call  # the backward runs the layers last to first
    g, x = g.float()[0], x.float()[0]  # [T, D]
    D = g.size(-1)
    rms = x.square().mean(-1).sqrt()
    flat = g.abs().flatten()
    top = int(flat.nan_to_num(nan=float("inf")).argmax())
    t, c = divmod(top, D)
    low = int(rms.nan_to_num(nan=0.0).argmin())
    before = rms[t - 1].item() if t else float("nan")
    print(f"SMEAR_DEBUG layer={layer} g_nonfinite={int((~torch.isfinite(g)).sum())} "
          f"x_nonfinite={int((~torch.isfinite(x)).sum())} max|g|={flat[top].item():.3e} at t={t} c={c} "
          f"rms(x_t)={rms[t].item():.3e} rms(x_t-1)={before:.3e} min_rms(x)={rms[low].item():.3e} at t={low} "
          f"max|g|_there={g[low].abs().max().item():.3e} grad_w_finite={bool(torch.isfinite(grad_w).all())} "
          f"max|grad_w|={grad_w.abs().max().item():.3e}", flush=True)


@torch.library.custom_op("smear_debug::shift", mutates_args=())
def shift(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x + w * F.pad(x[:, :-1], (0, 0, 1, 0))


@shift.register_fake
def _(x, w):
    return torch.empty_like(x)


@torch.library.custom_op("smear_debug::shift_backward", mutates_args=())
def shift_backward(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad_w = (g[:, 1:].float() * x[:, :-1].float()).sum(dim=(0, 1))
    report(g, x, grad_w)
    grad_x = g + F.pad((w * g)[:, 1:], (0, 0, 0, 1))
    return grad_x, grad_w.to(w.dtype)


@shift_backward.register_fake
def _(g, x, w):
    return torch.empty_like(g), torch.empty_like(w)


def _setup(ctx, inputs, output):
    x, w = inputs
    ctx.save_for_backward(x, w)


def _backward(ctx, grad):
    x, w = ctx.saved_tensors
    return shift_backward(grad, x, w)


shift.register_autograd(_backward, setup_context=_setup)


def main():
    import check_rtx5090 as smoke
    os.chdir(ROOT)
    args = argparse.Namespace(seed=999, val_seq_len=65536)
    smoke.configure(args)  # the modules read the environment at import
    import track_1_short.model.gpt as gpt
    gpt.token_shift = shift  # what the forward calls; opaque to the compiler
    try:
        smoke.run_check(args)
        print("SMEAR_DEBUG smoke=pass: with the opaque op the gradient is finite, so the compiled shift was at fault")
    except Exception as exc:
        print(f"SMEAR_DEBUG smoke=fail: {exc}")


if __name__ == "__main__":
    main()

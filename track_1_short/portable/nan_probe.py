"""RTX_NAN_PROBE=1: the first non-finite value a run produces, and where it appeared.

A NaN in a printed loss says nothing about its origin: by the time it is printed the weights may
have been NaN for twenty steps, and the warmup prints no loss at all. This walks the three places
one step can first go non-finite -- the microbatch loss, the gradients its backward left (including
the n-gram sink's, which is the port's own path), and the parameters and fp32 scale buffers the
optimizer step and the fp8 refresh wrote -- and reports the first one, once, then goes quiet.

Enabled per run and read once at import, so a normal run pays one `if` per microbatch. The hooks
call .item() and reduce over every parameter and buffer (the n-gram table among them), so a probe
run is not a timing run.

The cheapest probe is the memory preflight: it runs the whole warmup and nothing on the clock.

    RTX_VAL_SEQ_LEN=65536 TRAIN_SEED=999 RTX_PREFLIGHT=1 RTX_NAN_PROBE=1 bash run_rtx.sh
"""
import math
import os

import torch
from torch import Tensor, nn

ENABLED = os.environ.get("RTX_NAN_PROBE") == "1"

# fp8 tensors are left out: torch.isfinite has no kernel for them, and every writer of an fp8 cache
# clamps into range, so a NaN there comes from the bf16 weights or the fp32 scales checked here.
_CHECKED = (torch.float64, torch.float32, torch.float16, torch.bfloat16)
# Above this many elements, counting the bad ones would allocate a mask the size of the tensor --
# 2 GB for the n-gram table, which the card does not have spare. Report the range instead.
_COUNT_LIMIT = 1 << 24


def _nonfinite(name: str, t: Tensor | None) -> str | None:
    """A report like `grad lm_head.weight (nan, 3/768 elements)`, or None if `t` is all finite."""
    if t is None or t.dtype not in _CHECKED or t.numel() == 0:
        return None
    # amin / amax are reductions, so this allocates nothing, and both NaN and +-inf propagate.
    lo, hi = float(t.amin()), float(t.amax())
    if math.isfinite(lo) and math.isfinite(hi):
        return None
    kind = "nan" if (math.isnan(lo) or math.isnan(hi)) else "inf"
    if t.numel() > _COUNT_LIMIT:
        return f"{name} ({kind}, min {lo} max {hi}, {t.numel()} elements)"
    bad = int((~torch.isfinite(t)).sum())
    return f"{name} ({kind}, {bad}/{t.numel()} elements)"


class NanProbe:
    """Reports where the first non-finite value appeared. `model` is the uncompiled module."""

    def __init__(self, model: nn.Module, print0):
        self.model = model
        self.print0 = print0
        self.tripped = False

    def _first(self, named) -> str | None:
        for name, t in named:
            if (report := _nonfinite(name, t)) is not None:
                return report
        return None

    def _report(self, where: str, detail: str):
        self.tripped = True
        self.print0(f"NAN PROBE: first non-finite at {where} -- {detail}", console=True)

    def state(self, where: str):
        """Every parameter and buffer: call wherever the state is supposed to be clean."""
        if self.tripped:
            return
        detail = (self._first((f"param {n}", p) for n, p in self.model.named_parameters())
                  or self._first((f"buffer {n}", b) for n, b in self.model.named_buffers()))
        if detail is not None:
            self._report(where, detail)

    def after_backward(self, step: int, index: int, loss: Tensor, sink: Tensor):
        """One microbatch's loss and the gradients its backward left. Call before the gradients are
        folded into the accumulators, which drops every .grad."""
        if self.tripped:
            return
        value = loss.item()
        where = f"step {step} microbatch {index}"
        self.print0(f"NAN PROBE: {where} loss {value}", console=True)
        if not math.isfinite(value):
            self._report(where, f"the loss itself ({value}) -- the forward, not the update")
            return
        detail = (self._first([("grad ngram_sink", sink.grad)])
                  or self._first((f"grad {n}", p.grad) for n, p in self.model.named_parameters()))
        if detail is not None:
            self._report(where, detail)

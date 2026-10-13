"""Gradient accumulation over a step's microbatches, in fp32, handed to the optimizer as the record's
rank-averaged gradient.

The record: each rank backpropagates loss.sum() over its microbatch, the optimizer averages the ranks
(AVG reduce-scatter / all-reduce), and the Adam parameters sum two steps' gradients before their update
on odd steps (StepGraphs.backward). Here: every microbatch's .grad is added into an fp32 accumulator
right after its backward (so bf16 rounding does not compound over up to 24 microbatches); at the step's
end the ANVIL banks get sum / VIRTUAL_WORLD, and on Adam steps the Adam parameters get their two-step
sum / VIRTUAL_WORLD -- the record's numbers, with the rank average replaced by an exact power-of-two scale.

A parameter that received no gradient during its window is left without one, as in the record, where
the optimizer then skips it.
"""
import torch
from torch import nn

from track_1_short.portable.config import VIRTUAL_WORLD

# The row-sparse tables never get a .grad (portable/tables.py owns their gradients).
SPARSE_LABELS = frozenset({"value_embeds"})


class GradAccumulator:
    def __init__(self, param_cfgs: dict):
        self.params = [p for p, cfg in param_cfgs.items() if cfg.label not in SPARSE_LABELS]
        self.anvil = [p for p in self.params if param_cfgs[p].optim == "anvil"]
        self.adam = [p for p in self.params if param_cfgs[p].optim == "adam"]
        self.acc = {p: torch.zeros_like(p, dtype=torch.float32) for p in self.params}
        self.seen: set[nn.Parameter] = set()

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.acc.values())

    def reset(self):
        for t in self.acc.values():
            t.zero_()
        self.seen.clear()
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def collect(self):
        """After one microbatch's backward: fold every .grad into its accumulator and drop it."""
        live = [p for p in self.params if p.grad is not None]
        if live:
            torch._foreach_add_([self.acc[p] for p in live], [p.grad for p in live])
        for p in live:
            p.grad = None
        self.seen.update(live)

    @torch.no_grad()
    def _release(self, params):
        for p in params:
            if p in self.seen:
                acc = self.acc[p]
                p.grad = (acc * (1.0 / VIRTUAL_WORLD)).to(p.dtype)
                acc.zero_()
                self.seen.discard(p)
            else:
                p.grad = None

    def release(self, do_adam: bool):
        """Before the optimizer step: the banks' gradient every step, the Adam parameters' on Adam steps
        (their accumulators keep summing across the even step otherwise)."""
        self._release(self.anvil)
        if do_adam:
            self._release(self.adam)
        else:
            for p in self.adam:
                p.grad = None

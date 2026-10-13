"""The two row-sparse tables of the record on one GPU: the hashed n-gram table and value_embeds.

The record shards both across 8 ranks and moves rows between them (ngram_table.py, perf/value_embed_pull.py,
perf/row_prefetch.py). With one process owning every row, all of that collapses:

n-gram table (LocalNgramTable)
  - The whole table sits on the GPU, at 1/RTX_NGRAM_DIVISOR of the record's rows (config.py). The model's
    `ngram_cache` buffer IS the table, so the forward reads it at the row ids themselves: no cycle cache,
    no pulls, no want lists. That is exact for the same reason the record's cache is: the table only
    changes at its update events, so every forward of a cycle reads the same rows either way.
  - Each microbatch's row gradient (the grad of the zero "sink" leaf, [2T, 768]) is scatter-added into a
    table-sized fp16 gradient right after its backward, so nothing per microbatch is held until the event.
  - The event runs the record's own row-Adam kernel (perf/kernels/ngram_adam.py) over EVERY row, with a
    zero gradient where no token touched it. With beta1 = 0 a zero gradient gives a zero update and the
    cautious decay is gated on the update, so untouched weights do not move and their second moment
    takes this event's beta2 now instead of on its next touch -- the same multiplies in the same order
    as the record's lazy replay, so the same numbers. One dense pass (~10 ms on a 5090) per event.
  - grad_mul / sq_mul average over the record's 8 ranks (config.VIRTUAL_WORLD), as the record's do.

value_embeds (LocalValueEmbeds)
  - The backward already adds every microbatch's rows into one persistent fp16 buffer (grad_accum,
    perf/kernels/value_embed.py), so at the event that buffer IS the cycle's summed gradient. It is
    scaled by 1/8 (the record's AVG reduce-scatter) and Adam runs on the whole parameter -- one rank's
    shard is the whole table -- exactly as ValueEmbedPull.adam_update does on a shard.

LocalTables duck-types RowPrefetch.optimizer_event for TrainingManager.step_optimizers, and its event
duck-types optim/anvil.py SparseUpdate (launch / update / finish).
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from track_1_short.ngram_table import (
    GRAD_ACCUM_DTYPE,
    NGRAM_ADAM_BETA2,
    NGRAM_DIM,
    NGRAM_LR_MUL,
    NGRAM_VOCAB_SIZE,
    adam_beta2_and_wd_mul,
)
from track_1_short.optim.anvil import AnvilAndAdam
from track_1_short.perf.kernels.ngram_adam import adam_rows_
from track_1_short.perf.kernels.row_scatter import scatter_add_rows
from track_1_short.perf.kernels.value_embed import GRAD_DTYPE
from track_1_short.portable.config import VIRTUAL_WORLD

assert 0 < NGRAM_ADAM_BETA2 < 1


class LocalNgramTable:
    """The whole n-gram table, its fp16 gradient and its beta1 = 0 row-RMS Adam state, on one GPU."""

    def __init__(self, max_events: int, device: torch.device):
        rows = NGRAM_VOCAB_SIZE
        self.device = device
        self.table = torch.zeros(rows, NGRAM_DIM, dtype=torch.bfloat16, device=device)
        self.grad = torch.zeros(rows, NGRAM_DIM, dtype=GRAD_ACCUM_DTYPE, device=device)
        self.exp_avg_sq = torch.zeros(rows, dtype=torch.float32, device=device)
        self.last_event = torch.zeros(rows, dtype=torch.int32, device=device)
        self.beta2_history = torch.zeros(max_events + 1, dtype=torch.float32, device=device)
        # Every row is its own entry, so every program of the row-Adam kernel commits its row.
        self.all_rows = torch.arange(rows, dtype=torch.int32, device=device)
        self.sinks: dict[int, Tensor] = {}
        self.reset()

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in (self.table, self.grad, self.exp_avg_sq, self.last_event, self.all_rows))

    def reset(self):
        """Back to the initial state, in place (the post-warmup reset). The record also starts at zero."""
        self.table.zero_()
        self.grad.zero_()
        self.exp_avg_sq.zero_()
        self.last_event.zero_()
        self.adam_events = 0
        self.pending_grads = 0

    def grad_sink(self, num_tokens: int) -> Tensor:
        """The zeros leaf the forward adds to the looked-up rows; its .grad is the rows' gradient."""
        if num_tokens not in self.sinks:
            self.sinks[num_tokens] = torch.zeros(2 * num_tokens, NGRAM_DIM, dtype=torch.bfloat16,
                                                 device=self.device, requires_grad=True)
        sink = self.sinks[num_tokens]
        sink.grad = None
        return sink

    @torch.no_grad()
    def accumulate_grad(self, row_ids: Tensor, sink_grad: Tensor):
        """Add one microbatch's row gradient ([2T, D], rows `row_ids`) into the table-sized gradient."""
        assert row_ids.dtype == torch.int32 and row_ids.shape[0] == sink_grad.shape[0]
        scatter_add_rows(sink_grad.contiguous(), row_ids, self.grad)
        self.pending_grads += 1

    @torch.no_grad()
    def adam_update(self, step: int, lr: float, weight_decay: float, eps: float):
        """One Adam event over every row (see the module docstring); `lr`, `weight_decay` are Adam's
        scheduled base values, as NgramTable.adam_update takes them."""
        assert self.pending_grads, "Adam event with no gradient since the last one"
        beta2, wd_mul = adam_beta2_and_wd_mul(step)
        lr = lr * NGRAM_LR_MUL
        self.adam_events += 1
        event = self.adam_events
        self.beta2_history[event].fill_(beta2)
        adam_rows_(self.table, self.exp_avg_sq, self.last_event, self.beta2_history, self.grad, self.all_rows,
                   self.all_rows, event, beta2=beta2, eps=eps,
                   step_size=lr * (1 - beta2 ** event) ** 0.5,  # beta1 = 0: no first-moment bias
                   decay=lr * lr * weight_decay * wd_mul,
                   grad_mul=1 / VIRTUAL_WORLD, sq_mul=(1 - beta2) / VIRTUAL_WORLD ** 2)
        self.grad.zero_()
        self.pending_grads = 0


class LocalValueEmbeds:
    """value_embeds' gradient buffer and its update event, with one process owning every row."""

    def __init__(self, param: nn.Parameter, optimizer: AnvilAndAdam, device: torch.device):
        self.param = param
        self.optimizer = optimizer
        # 309 MB fp16: the live cycle's gradient, written by the backward (perf/kernels/value_embed.py).
        self.grad_accum = torch.zeros(param.shape, dtype=GRAD_DTYPE, device=device)

    def reset(self):
        self.grad_accum.zero_()

    @torch.no_grad()
    def adam_update(self):
        assert self.param.grad is None, "value_embeds got a dense gradient"
        grad = self.grad_accum.mul_(1.0 / VIRTUAL_WORLD)  # exact in fp16: a power of two
        self.optimizer.adam_update_shard(self.param, grad)
        grad.zero_()


@dataclass(slots=True)
class LocalTableEvent:
    """Both tables' updates at their place in AnvilAndAdam.step (optim/anvil.py SparseUpdate)."""
    ngram: LocalNgramTable
    value_embeds: LocalValueEmbeds
    step: int
    lr: float
    weight_decay: float
    eps: float

    def launch(self):
        pass  # nothing to exchange

    def update(self):
        self.value_embeds.adam_update()
        self.ngram.adam_update(self.step, self.lr, self.weight_decay, self.eps)

    def finish(self):
        pass


class LocalTables:
    """Stands in for RowPrefetch in TrainingManager.step_optimizers: the tables update on `is_update`."""

    def __init__(self, ngram: LocalNgramTable, value_embeds: LocalValueEmbeds, is_update):
        self.ngram = ngram
        self.value_embeds = value_embeds
        self.is_update = is_update

    def optimizer_event(self, step: int, lr: float, weight_decay: float, eps: float) -> LocalTableEvent | None:
        if not self.is_update(step):
            return None
        return LocalTableEvent(self.ngram, self.value_embeds, step, lr, weight_decay, eps)

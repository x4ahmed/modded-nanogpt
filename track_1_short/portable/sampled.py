"""The sampled-softmax candidate sets (sampled_softmax.py) for sequential microbatches.

The record builds one candidate set per rank per step, from that rank's microbatch, on a prep thread
and a side copy stream, and relies on exactly one microbatch per step (its loss keeps no copy of the
gathered rows). Here each microbatch plays one of those ranks: its set is built from its own targets
by the record's CandidateBuilder right before its forward and uploaded through a ring of pinned host
slots (the record's CandidateStaging ring, without its prep thread and copy stream) as async copies on
the compute stream. A plain copy_ from pageable memory synchronizes the stream: the host would wait for
the device to drain before every forward, and the device would idle while the host launched it.
Microbatches run forward and backward one after another on that one stream, so a set's device buffers
are never rewritten before its backward has read them -- the record's race contract, kept by stream order.

The negatives come from one stride sweep that advances with every build (the record staggers one
sweep per rank instead): consecutive microbatches get disjoint negative windows, and across a run every
class is visited uniformly, which is what the staggering is for.
"""
import numpy as np
import torch
from torch import Tensor

from track_1_short.perf.kernels.transpose import transpose_copy
from track_1_short.perf.sampled_softmax_overlap import PINNED_SLOTS, PinnedSlot
from track_1_short.sampled_softmax import CandidateBuilder, SampledLoss, _stage_index, candidate_count_at
from track_1_short.schedule import TrainingSchedule


class PortableSampledSoftmax:
    def __init__(self, schedule: TrainingSchedule, vocab_size: int, model_dim: int, max_rows: int,
                 device: torch.device):
        self.schedule = schedule
        self.counts = [candidate_count_at(schedule, s) for s in range(schedule.total_steps + 1)]
        p_values = sorted(set(self.counts) - {0})
        assert p_values
        max_p = max(p_values)
        self.builder = CandidateBuilder(vocab_size, max_p, max_rows)
        self.prefix_table: np.ndarray | None = None
        self.candidates = torch.zeros(max_p, dtype=torch.int64, device=device)
        target_pos = torch.zeros(max_rows, dtype=torch.int64, device=device)
        prefix_pos = torch.full((max_rows,), -1, dtype=torch.int64, device=device)
        vocab_pos = torch.empty(vocab_size, dtype=torch.int32, device=device)
        self.arange = torch.arange(max_p, dtype=torch.int32, device=device)
        e4m3 = torch.float8_e4m3fn
        self.loss_inputs = {
            p: SampledLoss(rows=torch.zeros(p, model_dim, dtype=e4m3, device=device),
                           rows_t=torch.zeros(model_dim, p, dtype=e4m3, device=device),
                           target_pos=target_pos, prefix_pos=prefix_pos, vocab_pos=vocab_pos)
            for p in p_values
        }
        # Each slot's event is recorded after its H2Ds; the host rewrites a slot only once they retired.
        pinned = lambda n: torch.empty(n, dtype=torch.int64, pin_memory=True)
        self.slots = [PinnedSlot(pinned(max_p), pinned(max_rows), pinned(max_rows), torch.cuda.Event())
                      for _ in range(PINNED_SLOTS)]
        self.next_slot = 0
        self.reset()

    def reset(self):
        self.builder.reset(rank=0, world_size=1)

    def set_prefix_table(self, prefix_table: np.ndarray):
        self.prefix_table = prefix_table

    def count_change_steps(self) -> list[int]:
        return [s for s in range(1, len(self.counts)) if self.counts[s] != self.counts[s - 1]]

    def stage_keys(self, steps) -> set[tuple[int, int]]:
        """(P, stage index) of each step: the compiled-graph variants the sampled loss adds."""
        return {(self.counts[s], _stage_index(self.schedule, s)) for s in steps}

    @torch.no_grad()
    def prepare(self, step: int, targets_cpu: Tensor, lm_head_f8_col: Tensor) -> SampledLoss | None:
        """This microbatch's candidate set, uploaded, with its lm_head rows gathered from the current fp8
        cache; None on full-softmax steps. Call after the step's fp8 refresh, right before the forward."""
        P = self.counts[step]
        if not P:
            return None
        inputs = self.loss_inputs[P]
        candidates, target_pos, prefix_pos = self.builder.build(P, targets_cpu.numpy(), self.prefix_table)
        slot = self.slots[self.next_slot]
        self.next_slot = (self.next_slot + 1) % PINNED_SLOTS
        slot.uploaded.synchronize()
        uploads = [(candidates, slot.candidates, self.candidates), (target_pos, slot.target_pos, inputs.target_pos)]
        if prefix_pos is not None:
            uploads.append((prefix_pos, slot.prefix_pos, inputs.prefix_pos))
        # The host copy into the slot frees the builder's scratch for the next build at once.
        for src, staged, dst in uploads:
            n = src.shape[0]
            staged[:n].copy_(torch.from_numpy(src))
            dst[:n].copy_(staged[:n], non_blocking=True)
        slot.uploaded.record()
        cand = self.candidates[:P]
        torch.index_select(lm_head_f8_col.T.view(torch.uint8), 0, cand, out=inputs.rows.view(torch.uint8))
        transpose_copy(inputs.rows.view(torch.uint8), inputs.rows_t.view(torch.uint8))
        inputs.vocab_pos.fill_(-1)
        inputs.vocab_pos.index_copy_(0, cand, self.arange[:P])
        return inputs

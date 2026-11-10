"""The sampled-softmax candidate sets (sampled_softmax.py) for sequential microbatches.

The record builds one candidate set per rank per step, from that rank's microbatch, on a prep thread
and a side copy stream, and relies on exactly one microbatch per step (its loss keeps no copy of the
gathered rows). Here each microbatch plays one of those ranks: its set is built from its own targets
by the record's CandidateBuilder and uploaded synchronously right before its forward. Microbatches run
forward and backward one after another, so a set's buffers are never rewritten before its backward has
read them -- the record's race contract, kept by construction.

The negatives come from one stride sweep that advances with every build (the record staggers one
sweep per rank instead): consecutive microbatches get disjoint negative windows, and across a run every
class is visited uniformly, which is what the staggering is for.

experiment/vocab-prune (portable/vocab_prune.py): with pruning mass 0 the record's full-softmax steps
run at width PRUNED_WIDTH over the ids seen as training targets so far; with mass m > 0 every step runs
at min(record width, ranked_width(m)) over the microbatch's targets plus the most frequent ids so far,
built on the device (no host builder, no upload).
"""
import numpy as np
import torch
from torch import Tensor

from track_1_short.perf.kernels.transpose import transpose_copy
from track_1_short.portable import config as rtx
from track_1_short.portable.vocab_prune import (
    PRUNED_WIDTH, RankedCandidates, SeenTargets, pruned_counts, ranked_counts, ranked_width,
)
from track_1_short.sampled_softmax import CandidateBuilder, SampledLoss, _stage_index, candidate_count_at
from track_1_short.schedule import TrainingSchedule


class PortableSampledSoftmax:
    def __init__(self, schedule: TrainingSchedule, vocab_size: int, model_dim: int, max_rows: int,
                 device: torch.device, mass: float | None = None):
        self.schedule = schedule
        self.mass = rtx.VOCAB_PRUNE_MASS if mass is None else mass
        record_counts = [candidate_count_at(schedule, s) for s in range(schedule.total_steps + 1)]
        # Exact mode: the record's full-softmax steps (count 0) become pruned steps at PRUNED_WIDTH.
        self.pruned_steps = frozenset(s for s, count in enumerate(record_counts) if not count)
        self.counts = (ranked_counts(record_counts, self.mass, vocab_size) if self.mass
                       else pruned_counts(record_counts))
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
        self.seen = SeenTargets(vocab_size, PRUNED_WIDTH, device)
        self.ranked = RankedCandidates(vocab_size, max_rows, device) if self.mass else None
        self.prefix_table_d: Tensor | None = None
        self.reset()

    def reset(self):
        self.builder.reset(rank=0, world_size=1)
        self.seen.reset()
        if self.ranked is not None:
            self.ranked.reset()
        # (step, set version) whose rows the pruned slab holds; None after a sampled build overwrote the
        # shared candidates / vocab_pos / prefix_pos buffers (warmup interleaves the two kinds of step).
        self.pruned_loaded: tuple[int, int] | None = None

    def set_prefix_table(self, prefix_table: np.ndarray):
        self.prefix_table = prefix_table
        self.prefix_table_d = torch.from_numpy(prefix_table).to(self.candidates.device)

    def describe(self) -> str:
        """The run's pruning setup, one log line."""
        if not self.mass:
            steps = sorted(self.pruned_steps & set(range(self.schedule.total_steps)))
            return (f"mode=exact mass=0 width={PRUNED_WIDTH} on full-softmax steps {steps[0]}-{steps[-1]} "
                    f"({len(steps)} steps); candidates = ids seen as training targets so far")
        by_stage = {}
        for s in range(self.schedule.total_steps):
            by_stage.setdefault(_stage_index(self.schedule, s), []).append(self.counts[s])
        widths = "; ".join(f"stage {i}: " + "/".join(map(str, dict.fromkeys(w))) for i, w in by_stage.items())
        return (f"mode=ranked mass={self.mass} width={ranked_width(self.mass)} on every step ({widths}); "
                f"candidates = microbatch targets + most frequent ids so far")

    def summary(self) -> str:
        """End-of-run counters (syncs): call off the clock."""
        return self.ranked.summary() if self.ranked is not None else self.seen.summary()

    def count_change_steps(self) -> list[int]:
        return [s for s in range(1, len(self.counts)) if self.counts[s] != self.counts[s - 1]]

    def stage_keys(self, steps) -> set[tuple[int, int]]:
        """(P, stage index) of each step: the compiled-graph variants the sampled loss adds."""
        return {(self.counts[s], _stage_index(self.schedule, s)) for s in steps}

    @torch.no_grad()
    def prepare(self, step: int, targets_cpu: Tensor, lm_head_f8_col: Tensor,
                targets: Tensor | None = None) -> SampledLoss | None:
        """This microbatch's candidate set, uploaded, with its lm_head rows gathered from the current fp8
        cache; None if a pruned step falls back to the full softmax. Call after the step's fp8 refresh,
        right before the forward. `targets` is the device copy of targets_cpu (copied here if omitted)."""
        if targets is None:
            targets = targets_cpu.to(self.candidates.device)
        if self.ranked is not None:
            return self._prepare_ranked(step, targets, lm_head_f8_col)
        self.seen.mark(targets)
        if step in self.pruned_steps:
            return self._prepare_pruned(step, targets_cpu, targets, lm_head_f8_col)
        self.pruned_loaded = None
        P = self.counts[step]
        inputs = self.loss_inputs[P]
        candidates, target_pos, prefix_pos = self.builder.build(P, targets_cpu.numpy(), self.prefix_table)
        # Pageable H2D copies return once the host source is read, so the builder may reuse its scratch.
        self.candidates[:P].copy_(torch.from_numpy(candidates))
        inputs.target_pos[:target_pos.shape[0]].copy_(torch.from_numpy(target_pos))
        if prefix_pos is not None:
            inputs.prefix_pos[:prefix_pos.shape[0]].copy_(torch.from_numpy(prefix_pos))
        cand = self.candidates[:P]
        torch.index_select(lm_head_f8_col.T.view(torch.uint8), 0, cand, out=inputs.rows.view(torch.uint8))
        transpose_copy(inputs.rows.view(torch.uint8), inputs.rows_t.view(torch.uint8))
        inputs.vocab_pos.fill_(-1)
        inputs.vocab_pos.index_copy_(0, cand, self.arange[:P])
        return inputs

    def _prepare_pruned(self, step: int, targets_cpu: Tensor, targets: Tensor, lm_head_f8_col: Tensor):
        """C = every id seen as a target so far, filled to PRUNED_WIDTH; target positions looked up on the
        device. The set, its maps and the row slab are rebuilt only when needed (see reset)."""
        P = PRUNED_WIDTH
        inputs = self.loss_inputs[P]
        state = self.seen.refresh(targets_cpu.numpy())
        if state == "full":
            return None
        if state == "new" or self.pruned_loaded is None:
            self.candidates[:P].copy_(torch.from_numpy(self.seen.candidates))
            inputs.vocab_pos.fill_(-1)
            inputs.vocab_pos.index_copy_(0, self.candidates[:P], self.arange[:P])
            inputs.prefix_pos.fill_(-1)  # the prefix weight is 0 in these stages; -1 is the kernel's no-op
        if self.pruned_loaded != (step, self.seen.version):
            # The fp8 lm_head cache changes only between steps: one row gather per step (or per new set).
            torch.index_select(lm_head_f8_col.T.view(torch.uint8), 0, self.candidates[:P],
                               out=inputs.rows.view(torch.uint8))
            transpose_copy(inputs.rows.view(torch.uint8), inputs.rows_t.view(torch.uint8))
            self.pruned_loaded = (step, self.seen.version)
        n = targets.shape[0]
        inputs.target_pos[:n].copy_(inputs.vocab_pos.index_select(0, targets))
        return inputs

    def _prepare_ranked(self, step: int, targets: Tensor, lm_head_f8_col: Tensor) -> SampledLoss:
        """Ranked mode: the set, its row slab, and the target / prefix positions, all on the device."""
        P = self.counts[step]
        inputs = self.loss_inputs[P]
        cand = self.candidates[:P]
        cand.copy_(self.ranked.build(step, targets, P))
        torch.index_select(lm_head_f8_col.T.view(torch.uint8), 0, cand, out=inputs.rows.view(torch.uint8))
        transpose_copy(inputs.rows.view(torch.uint8), inputs.rows_t.view(torch.uint8))
        inputs.vocab_pos.fill_(-1)
        inputs.vocab_pos.index_copy_(0, cand, self.arange[:P])
        n = targets.shape[0]
        # Every target is a candidate (overflow is counted); the clamp only keeps an overflow in bounds.
        inputs.target_pos[:n].copy_(inputs.vocab_pos.index_select(0, targets).clamp_(min=0))
        if self.prefix_table_d is None:
            inputs.prefix_pos[:n].fill_(-1)
        else:
            # As the record: a prefix target outside the set is dropped (-1, the kernel's no-op).
            prefix = self.prefix_table_d.index_select(0, targets)
            pos = inputs.vocab_pos.index_select(0, prefix.clamp(min=0))
            inputs.prefix_pos[:n].copy_(torch.where(prefix >= 0, pos, -1))
        return inputs

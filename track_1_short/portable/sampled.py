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
built on the device (no host builder, no upload). RTX_VOCAB_PRUNE_EXACT_END=1 keeps the record's
full softmax on its full-softmax steps (prepare returns None there).
"""
import numpy as np
import torch
from torch import Tensor

from track_1_short.perf.kernels.transpose import transpose_copy
from track_1_short.portable import config as rtx
from track_1_short.portable.vocab_prune import (
    PRUNED_WIDTH, RankedCandidates, SeenTargets, pruned_counts, ranked_counts, ranked_width, setting_at,
)
from track_1_short.sampled_softmax import CandidateBuilder, SampledLoss, _stage_index, candidate_count_at
from track_1_short.schedule import TrainingSchedule


class PortableSampledSoftmax:
    def __init__(self, schedule: TrainingSchedule, vocab_size: int, model_dim: int, max_rows: int,
                 device: torch.device, mass: float | None = None, tail: int | None = None,
                 exact_end: bool | None = None, first: int | None = None, until: int | None = None,
                 switches: tuple | None = None, power: float | None = None, draw: str | None = None):
        self.schedule = schedule
        self.mass = rtx.VOCAB_PRUNE_MASS if mass is None else mass
        self.tail = (rtx.VOCAB_PRUNE_TAIL if tail is None else tail) if self.mass else 0
        self.exact_end = bool(rtx.VOCAB_PRUNE_EXACT_END if exact_end is None else exact_end) and bool(self.mass)
        record_counts = [candidate_count_at(schedule, s) for s in range(schedule.total_steps + 1)]
        # Stage gate (ranked modes): only steps first <= step < until are pruned; the rest run the record.
        self.first = (rtx.VOCAB_PRUNE_FROM_STEP if first is None else first) if self.mass else 0
        self.until = ((rtx.VOCAB_PRUNE_UNTIL_STEP if until is None else until) if self.mass else 0) or len(record_counts)
        self.gated = self.first > 0 or self.until < len(record_counts)
        # Exact mode: the record's full-softmax steps (count 0) become pruned steps at PRUNED_WIDTH.
        self.pruned_steps = frozenset(s for s, count in enumerate(record_counts) if not count)
        # Per-stage mass (stratified mode): (start, mass, tail) switches; the tail in force at each step.
        self.switches = tuple(rtx.VOCAB_PRUNE_SWITCH if switches is None else switches) if self.tail else ()
        self.tails = [setting_at(s, self.mass, self.tail, self.switches)[1] for s in range(len(record_counts))]
        self.counts = (ranked_counts(record_counts, self.mass, vocab_size, self.tail, self.exact_end,
                                     self.first, self.until, self.switches) if self.mass
                       else pruned_counts(record_counts))
        # Widths the stratified (weighted) kernel runs at: only these carry column weights.
        weighted = ({c for s, c in enumerate(self.counts) if c and self.first <= s < self.until}
                    if self.tail else set())
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
                           target_pos=target_pos, prefix_pos=prefix_pos, vocab_pos=vocab_pos,
                           col_logw=torch.zeros(p, dtype=torch.float32, device=device) if p in weighted else None)
            for p in p_values
        }
        self.seen = SeenTargets(vocab_size, PRUNED_WIDTH, device)
        self.power = (rtx.VOCAB_PRUNE_TAIL_POWER if power is None else power) if self.tail else 1.0
        self.draw = (rtx.VOCAB_PRUNE_TAIL_DRAW if draw is None else draw) if self.tail else "uniform"
        self.ranked = (RankedCandidates(vocab_size, max_rows, device, self.tail, self.power, self.draw)
                       if self.mass else None)
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
        where = "on every step"
        if self.exact_end:
            full = sorted(self.pruned_steps & set(range(self.schedule.total_steps)))
            where = f"on sampled steps, exact_end=1: full softmax on steps {full[0]}-{full[-1]} ({len(full)} steps)"
        if self.gated:
            where += (f", stage gate: pruned only on steps {self.first}-{min(self.until, self.schedule.total_steps) - 1},"
                      f" the record's own sets elsewhere")
        for start, mass, tail in self.switches:
            where += f", from step {start}: mass={mass} tail={tail} width={ranked_width(mass) + tail}"
        if self.tail:
            return (f"mode=stratified mass={self.mass} tail={self.tail} width={ranked_width(self.mass) + self.tail} "
                    f"{where} ({widths}); candidates = microbatch targets + most frequent ids so far + "
                    f"{self.tail} uniform tail ids weighted rest/{self.tail}"
                    + (f" to the power {self.power}" if self.power != 1 else "")
                    + (f"; tail draw={self.draw}" if self.draw != "uniform" else ""))
        return (f"mode=ranked mass={self.mass} width={ranked_width(self.mass)} {where} ({widths}); "
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
            if not self.counts[step]:
                return None  # exact_end or outside the stage gate: the record's full softmax on this step
            if not self.first <= step < self.until:
                self.ranked.observe(targets)  # stage gate: the record's own set; the ranking keeps counting
                return self._prepare_record(step, targets_cpu, lm_head_f8_col)
            return self._prepare_ranked(step, targets, lm_head_f8_col)
        self.seen.mark(targets)
        if step in self.pruned_steps:
            return self._prepare_pruned(step, targets_cpu, targets, lm_head_f8_col)
        self.pruned_loaded = None
        return self._prepare_record(step, targets_cpu, lm_head_f8_col)

    def _prepare_record(self, step: int, targets_cpu: Tensor, lm_head_f8_col: Tensor) -> SampledLoss:
        """The record's set: the microbatch's targets plus stride-sweep negatives, built on the host."""
        P = self.counts[step]
        inputs = self.loss_inputs[P]
        if inputs.col_logw is not None:
            inputs.col_logw.zero_()  # a width shared with pruned steps: weight 1 everywhere = the record's loss
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
        candidates, col_logw = self.ranked.build(step, targets, P, self.tails[step])
        cand.copy_(candidates)
        if col_logw is not None:
            inputs.col_logw.copy_(col_logw)
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
            # As the record: a prefix target outside the set is dropped (-1, the kernel's no-op); with a
            # tail, also one that is only a (weighted) tail column: prefix targets stay exact.
            prefix = self.prefix_table_d.index_select(0, targets)
            safe = prefix.clamp(min=0)
            pos = inputs.vocab_pos.index_select(0, safe)
            inputs.prefix_pos[:n].copy_(torch.where((prefix >= 0) & self.ranked.exact[safe], pos, -1))
        return inputs

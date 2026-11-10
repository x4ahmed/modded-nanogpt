"""Dynamic vocabulary pruning of the full-softmax steps (Phase 2 idea 2; docs/vocab-prune.md).

What: from step 0, every microbatch's targets are marked "seen" on the GPU (one index_fill per
microbatch). The steps the record runs on the full softmax (1107-1193) instead normalize over a
candidate set C of PRUNED_WIDTH ids: every id that has been a training target so far -- so every target
of the microbatch is in C and its loss term is exact -- filled up with never-seen ids, lowest ids first
(the 47 padding ids and the highest never-seen ids are the ones left out). These steps run through the
sampled-softmax kernels at width P = PRUNED_WIDTH (SampledSoftcappedCrossEntropy). Only ids that no
training target has used so far leave the normalizer, and their lm_head rows get no gradient there.
Validation is unchanged: all 50,304 ids and the canonical mask.

Width: a compile-time constant (one CE kernel and one compiled graph per width). The census
(tools/token_census.py) found 49,989 distinct target ids over the whole run, so an exact set fits in
50,304 - 2 x 128 = 50,048 at the GEMMs' 128-column granularity and in nothing narrower: this is the
largest exact cut. If a run ever sees more than PRUNED_WIDTH distinct targets, a microbatch whose
targets fall outside the current set uses the full softmax instead (counted in `stats`; that graph then
compiles on the clock).

Cost kept on the clock: the per-microbatch mark; one device-to-host copy of the seen mask whenever C is
(re)built; a host membership check of each pruned microbatch's targets; one lm_head row gather and
transpose per step (the fp8 cache changes only between steps); the sampled path's lm_head-gradient
densify.
"""
import numpy as np
import torch

PRUNED_WIDTH = 50_048
assert PRUNED_WIDTH % 128 == 0


def pruned_counts(counts: list[int]) -> list[int]:
    """Per-step candidate counts with every full-softmax step (count 0) moved to PRUNED_WIDTH."""
    return [count if count else PRUNED_WIDTH for count in counts]


def build_candidate_set(seen: np.ndarray, width: int) -> np.ndarray | None:
    """Ascending ids: every seen id plus the lowest never-seen ids up to `width`; None if they do not fit."""
    ids = np.flatnonzero(seen)
    if ids.size > width:
        return None
    fill = np.flatnonzero(~seen)[:width - ids.size]
    return np.sort(np.concatenate((ids, fill)))


class SeenTargets:
    """The run's seen-target mask (device) and the current pruned candidate set (host)."""

    def __init__(self, vocab_size: int, width: int, device):
        self.width = width
        self.seen = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        self.reset()

    def reset(self):
        self.seen.zero_()
        self.candidates: np.ndarray | None = None
        self.in_set: np.ndarray | None = None  # host (V,) membership of `candidates`
        self.version = 0  # bumps whenever `candidates` changes
        self.stats = dict(builds=0, full_softmax_microbatches=0, seen_at_first_build=None, seen_at_last_build=None)

    def mark(self, targets: torch.Tensor):
        """Asynchronous: marks this microbatch's target ids as seen."""
        self.seen.index_fill_(0, targets, True)

    def refresh(self, targets_cpu) -> str:
        """For a pruned microbatch whose targets are already marked: "same" if the current set holds
        every target, "new" after rebuilding it from the seen mask, "full" if the seen ids do not fit."""
        targets = np.asarray(targets_cpu)
        if self.in_set is not None and self.in_set[targets].all():
            return "same"
        seen = self.seen.cpu().numpy()  # a sync, only at the first build and when a new id appears
        candidates = build_candidate_set(seen, self.width)
        if candidates is None:
            self.stats["full_softmax_microbatches"] += 1
            return "full"
        self.candidates = candidates
        self.in_set = np.zeros(seen.size, dtype=bool)
        self.in_set[candidates] = True
        self.version += 1
        count = int(seen.sum())
        self.stats["builds"] += 1
        if self.stats["seen_at_first_build"] is None:
            self.stats["seen_at_first_build"] = count
        self.stats["seen_at_last_build"] = count
        return "new"

    def summary(self) -> str:
        """One log line; reads the seen mask (a sync), so call it off the clock."""
        stats = dict(self.stats, seen_total=int(self.seen.sum().item()), width=self.width,
                     left_out=self.seen.numel() - self.width)
        return " ".join(f"{key}={value}" for key, value in stats.items())

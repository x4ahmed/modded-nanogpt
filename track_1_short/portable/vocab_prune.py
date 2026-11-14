"""Dynamic vocabulary pruning (Phase 2 idea 2; docs/vocab-prune.md). Two modes, chosen per run by
RTX_VOCAB_PRUNE_MASS (portable/config.py):

- mass 0, "exact" (SeenTargets below): only ids never seen as a training target leave the softmax, on
  the full-softmax steps only.
- mass m > 0, "ranked" (RankedCandidates below): on EVERY step the candidate set is the microbatch's
  targets plus the most frequent ids seen so far in this run, at width min(record width, ranked_width(m)),
  where ranked_width(m) is the number of ids that cover 1 - m of the training text (census table below).
  Every target stays a candidate, so its loss term is exact; only ids ranked below the cut that are not
  a target of the microbatch leave the normalizer. No fallback is needed. Validation is untouched.
  RESULT (2026-11-11): +0.36..+0.60 loss at every mass -- ids outside the set are never negatives.
- mass m > 0 with tail t > 0 (RTX_VOCAB_PRUNE_TAIL), "stratified": the fix. The set is the exact part
  above at width ranked_width(m) (targets + best-ranked ids), plus t ids drawn uniformly at random from
  ALL the remaining ids (the tail), at total width min(record width, ranked_width(m) + t). Each tail id
  carries the weight w = (number of remaining ids) / t: the loss normalizes over
  Z_hat = sum_exact exp(z) + w * sum_tail exp(z), an unbiased estimate of the full softmax's Z. Every id
  is drawn now and then, so every id is still pushed down, with the right strength on average. The CE
  kernel adds log w to the tail columns (portable/weighted_ce.py). Prefix targets outside the exact
  part are dropped, as the record drops prefix targets outside its set.
- stage gate (RTX_VOCAB_PRUNE_FROM_STEP / RTX_VOCAB_PRUNE_UNTIL_STEP, with a mass): only steps
  from <= step < until are pruned; every other step runs the record's own set and kernel (or its full
  softmax). The ranking still counts every microbatch's targets. Locates where the loss gap comes from.
- per-stage mass (RTX_VOCAB_PRUNE_SWITCH="step:mass:tail[,step:mass:tail...]", stratified mode): from
  each step on, that mass and tail replace the run's. Each setting's width has its own compiled kernel.

Exact mode in detail: from step 0, every microbatch's targets are marked "seen" on the GPU (one index_fill per
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

# Ranked mode. Ids that cover 1 - m of the run's training targets (tools/token_census.py, 2026-11-10,
# training data only; the loader is deterministic, so every run sees the same tokens): mass m -> ids.
KEPT_IDS_BY_MASS = {0.001: 46_223, 0.01: 36_968, 0.02: 30_912, 0.05: 19_977, 0.07: 15_532,
                    0.10: 11_038, 0.15: 6_576, 0.20: 4_062, 0.30: 1_623}
# Every target of a microbatch must be a candidate: at most 5,292 distinct targets per 16,384-token
# microbatch over the whole run (census, stage 1), so no width goes below 6,144.
MIN_RANKED_WIDTH = 6_144
RANKED_MICROBATCH_TOKENS = 16_384


def ranked_width(mass: float) -> int:
    """The candidate width of pruning mass `mass`: its kept ids, rounded up to 128, at least the floor."""
    kept = KEPT_IDS_BY_MASS[mass]
    return max(MIN_RANKED_WIDTH, -(-kept // 128) * 128)


def setting_at(step: int, mass: float, tail: int, switches=()) -> tuple[float, int]:
    """(mass, tail) in force at `step`: the last switch (start, mass, tail) with start <= step, else the base."""
    for start, switch_mass, switch_tail in switches:
        if start <= step:
            mass, tail = switch_mass, switch_tail
    return mass, tail


def ranked_counts(record_counts: list[int], mass: float, vocab_size: int, tail: int = 0,
                  exact_end: bool = False, first: int = 0, until: int = 0, switches=()) -> list[int]:
    """Per-step widths in ranked mode: the record's width (full softmax = vocab_size), capped at the
    mass's exact width plus the tail. exact_end keeps the record's full-softmax steps (count 0) as they are.
    Stage gate: only steps first <= step < until are pruned (until 0 = no limit); every other step keeps
    the record's count, so it runs the record's own set (or its full softmax, count 0).
    Per-stage mass: `switches` = (start, mass, tail) triples; from each start on, that mass and tail apply."""
    until = until or len(record_counts)
    counts = []
    for step, count in enumerate(record_counts):
        if not first <= step < until:
            counts.append(count)
        elif exact_end and not count:
            counts.append(0)
        else:
            step_mass, step_tail = setting_at(step, mass, tail, switches)
            counts.append(min(count or vocab_size, ranked_width(step_mass) + step_tail))
    return counts


def ranked_widths_in_use(record_widths, mass: float, tail: int, vocab_size: int, switches=()) -> tuple[int, ...]:
    """Every width ranked_counts can produce from these record widths (kernels compile per width)."""
    settings = [(mass, tail), *((m, t) for _, m, t in switches)]
    return tuple(sorted({w for m, t in settings for w in ranked_counts([*record_widths, 0], m, vocab_size, t)}))


def weighted_kernel_source(source: str) -> str:
    """The record's CE kernel source with a per-column log weight (col_logw) added to every exp():
    the forward sum, the dense gradient, and the sparse target correction. With col_logw == 0 it is the
    record's kernel; the record's own source string is not modified."""
    swaps = [
        ("    const float* __restrict__ prefix_weight_ptr,\n",
         "    const float* __restrict__ prefix_weight_ptr,\n    const float* __restrict__ col_logw,\n"),
        ("thread_sum += __expf(tmp - A);", "thread_sum += __expf(tmp - A + col_logw[idx + k]);"),
        ("          float sigmoid_u = __half2float(sigmoid_us[j + t]);\n"
         "          float z = A * sigmoid_u;\n"
         "          float p = __expf(z - lse);\n",
         "          float sigmoid_u = __half2float(sigmoid_us[j + t]);\n"
         "          float z = A * sigmoid_u;\n"
         "          float p = __expf(z - lse + col_logw[idx + j + t]);\n"),
        ("    float sigmoid_u = __half2float(smem[target]);\n"
         "    float z = A * sigmoid_u;\n"
         "    float p = __expf(z - lse);\n",
         "    float sigmoid_u = __half2float(smem[target]);\n"
         "    float z = A * sigmoid_u;\n"
         "    float p = __expf(z - lse + col_logw[target]);\n"),
    ]
    for old, new in swaps:
        assert source.count(old) == 1, f"CE kernel source changed: {old!r}"
        source = source.replace(old, new)
    return source


class RankedCandidates:
    """Ranked-mode candidate sets, built on the device with no host work and no sync.

    Per microbatch: count the targets (the ranking's data: this run's targets so far), then select every
    target plus the best-ranked non-targets up to `width`, ascending by id (the kernels' layout). The
    ranking (ids by count, descending) is recomputed once per step, from the counts before that step.
    """

    TAIL_SEED = 20011  # the tail's own generator: the global RNG streams are never touched

    def __init__(self, vocab_size: int, max_rows: int, device, tail: int = 0):
        self.vocab_size = vocab_size
        self.tail = tail
        self.counts = torch.zeros(vocab_size, dtype=torch.int32, device=device)
        self.is_target = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        self.extra = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        self.exact = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        self.tail_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        self.ones = torch.ones(max_rows, dtype=torch.int32, device=device)
        self.overflow = torch.zeros((), dtype=torch.int64, device=device)
        self.generator = torch.Generator(device=device)
        self.reset()

    def reset(self):
        self.counts.zero_()
        self.overflow.zero_()
        self.order = torch.arange(self.vocab_size, device=self.counts.device)
        self.ranked_step: int | None = None
        self.generator.manual_seed(self.TAIL_SEED)

    def observe(self, targets: torch.Tensor):
        """Stage gate: count a microbatch that runs the record's own set, so the ranking stays the
        run's targets so far."""
        self.counts.index_add_(0, targets, self.ones[:targets.numel()])

    def build(self, step: int, targets: torch.Tensor, width: int, tail: int | None = None):
        """(candidates, col_logw): (width,) int64 ids ascending -- all of `targets`, the best-ranked other
        ids up to width - tail, and `tail` uniform draws from the rest -- and their (width,) fp32 log
        weights (0 for exact ids, log(rest / tail) for tail ids; None without a tail). `tail` defaults to
        the run's tail; a per-stage switch passes its own."""
        tail = self.tail if tail is None else tail
        if self.ranked_step != step:
            self.order = torch.argsort(self.counts, descending=True, stable=True)
            self.ranked_step = step
        self.counts.index_add_(0, targets, self.ones[:targets.numel()])
        self.is_target.zero_()
        self.is_target.index_fill_(0, targets, True)
        n_targets = self.is_target.sum()
        budget = width - tail
        # Non-targets in rank order; keep the first budget - n_targets of them.
        free = ~self.is_target[self.order]
        take = free & (torch.cumsum(free, 0) <= budget - n_targets)
        self.extra.index_put_((self.order,), take)
        # More distinct targets than the exact budget cannot happen at the census floor; counted, never silent.
        self.overflow += n_targets > budget
        torch.logical_or(self.is_target, self.extra, out=self.exact)
        if not tail:
            return torch.argsort((~self.exact).to(torch.int8), stable=True)[:width], None
        # Uniform draw without replacement from the ids outside the exact part: the `tail` smallest keys.
        keys = torch.rand(self.vocab_size, generator=self.generator, device=self.counts.device)
        keys.masked_fill_(self.exact, 2.0)
        self.tail_mask.zero_()
        self.tail_mask.index_fill_(0, torch.topk(keys, tail, largest=False).indices, True)
        selected = self.exact | self.tail_mask
        candidates = torch.argsort((~selected).to(torch.int8), stable=True)[:width]
        rest = self.vocab_size - self.exact.sum()
        log_w = torch.log(rest.to(torch.float32) / tail)
        col_logw = torch.where(self.tail_mask[candidates], log_w, torch.zeros((), device=log_w.device))
        return candidates, col_logw

    def summary(self) -> str:
        """Reads the device counters (a sync): call off the clock."""
        return f"overflow_microbatches={int(self.overflow.item())} distinct_seen={int((self.counts > 0).sum().item())}"


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

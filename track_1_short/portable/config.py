"""Settings of the single-GPU port, from the environment (read once, at import).

PORTABLE=1 selects the port everywhere the record's modules branch on it. The other knobs only matter
under PORTABLE and default to values sized for one 32 GB RTX 5090:

  RTX_MICROBATCH_TOKENS  tokens per microbatch (default 16384). Every stage's batch must be a multiple:
                         the record's batches are 8/16/24/20/8 x 16384, so 16384 and 8192 both work.
  RTX_NGRAM_DIVISOR      the n-gram table keeps 84,602,880 / divisor rows (default 32 -> 2.64M rows,
                         4.1 GB in bf16 plus the same again for its gradient). Power of two, <= 2048.
  RTX_VAL_SEQ_LEN        tokens per validation forward (default 262144: what each of the record's 8
                         ranks reads, so the metric is computed exactly as the record computes it).
  RTX_EVAL_CE_SLAB_ROWS  rows per slab of the validation logits (default 4096; fp32 [rows, 50304]).
  RTX_WARMUP_MICROBATCHES  microbatches per warmup step (default 2: compiles every graph and exercises
                         the accumulation path; the warmup is untimed and its state is discarded).
  RTX_MAX_STEPS          stop after this many optimizer steps (smoke tests); the run then validates.
  RTX_VOCAB_PRUNE_MASS   experiment/vocab-prune: 0 (default) = exact pruning of never-seen ids on the
                         full-softmax steps; m in vocab_prune.KEPT_IDS_BY_MASS = ranked pruning on every
                         step at the width covering 1 - m of the training text (portable/vocab_prune.py).
  RTX_VOCAB_PRUNE_TAIL   with a mass: t uniform, reweighted tail ids per set (stratified mode; 0 = off).
  RTX_VOCAB_PRUNE_EXACT_END  1 = with a mass, keep the record's full softmax on its full-softmax steps
                         (1107-1193); only the sampled steps are pruned.
  RTX_VOCAB_PRUNE_FROM_STEP / RTX_VOCAB_PRUNE_UNTIL_STEP  stage gate, with a mass: prune only steps
                         from <= step < until (defaults 0 and 0 = no limit); the other steps run the record.
  RTX_VOCAB_PRUNE_SWITCH  per-stage mass, stratified mode: "step:mass:tail[,step:mass:tail...]"; from each
                         step on, that mass and tail replace RTX_VOCAB_PRUNE_MASS / _TAIL.
  RTX_VOCAB_PRUNE_TAIL_POWER  b in [0, 1] (default 1): a tail id weighs (rest / tail) ** b; b < 1 = smaller
                         gradient spikes, biased (portable/vocab_prune.py).
"""
import os

PORTABLE = os.environ.get("PORTABLE") == "1"

# The record's world: gradients are averaged over 8 ranks, so a step's summed microbatch gradients
# are scaled by 1/8, whatever the number of microbatches.
VIRTUAL_WORLD = 8

MICROBATCH_TOKENS = int(os.environ.get("RTX_MICROBATCH_TOKENS", "16384"))
NGRAM_DIVISOR = int(os.environ.get("RTX_NGRAM_DIVISOR", "32"))
VAL_SEQ_LEN = int(os.environ.get("RTX_VAL_SEQ_LEN", "262144"))
EVAL_CE_SLAB_ROWS = int(os.environ.get("RTX_EVAL_CE_SLAB_ROWS", "4096"))
WARMUP_MICROBATCHES = int(os.environ.get("RTX_WARMUP_MICROBATCHES", "2"))
MAX_STEPS = int(os.environ["RTX_MAX_STEPS"]) if os.environ.get("RTX_MAX_STEPS") else None
VOCAB_PRUNE_MASS = float(os.environ.get("RTX_VOCAB_PRUNE_MASS") or "0")
VOCAB_PRUNE_TAIL = int(os.environ.get("RTX_VOCAB_PRUNE_TAIL") or "0")
VOCAB_PRUNE_EXACT_END = os.environ.get("RTX_VOCAB_PRUNE_EXACT_END", "0") == "1"
VOCAB_PRUNE_FROM_STEP = int(os.environ.get("RTX_VOCAB_PRUNE_FROM_STEP") or "0")
VOCAB_PRUNE_UNTIL_STEP = int(os.environ.get("RTX_VOCAB_PRUNE_UNTIL_STEP") or "0")


def _parse_switch(text: str) -> tuple[tuple[int, float, int], ...]:
    """RTX_VOCAB_PRUNE_SWITCH -> ((step, mass, tail), ...)."""
    switches = []
    for item in filter(None, text.split(",")):
        parts = item.split(":")
        assert len(parts) == 3, "RTX_VOCAB_PRUNE_SWITCH: step:mass:tail[,step:mass:tail...]"
        switches.append((int(parts[0]), float(parts[1]), int(parts[2])))
    return tuple(switches)


VOCAB_PRUNE_SWITCH = _parse_switch(os.environ.get("RTX_VOCAB_PRUNE_SWITCH") or "")
VOCAB_PRUNE_TAIL_POWER = float(os.environ.get("RTX_VOCAB_PRUNE_TAIL_POWER") or "1")

assert MICROBATCH_TOKENS % 2048 == 0, "microbatches are whole 2048-token units"
assert NGRAM_DIVISOR >= 1 and NGRAM_DIVISOR & (NGRAM_DIVISOR - 1) == 0 and NGRAM_DIVISOR <= 2048
assert VAL_SEQ_LEN % 128 == 0, "validation forwards are whole attention blocks"
assert EVAL_CE_SLAB_ROWS > 0 and WARMUP_MICROBATCHES >= 1
assert MAX_STEPS is None or MAX_STEPS >= 1
if VOCAB_PRUNE_MASS:
    from track_1_short.portable.vocab_prune import KEPT_IDS_BY_MASS, RANKED_MICROBATCH_TOKENS
    if VOCAB_PRUNE_MASS not in KEPT_IDS_BY_MASS:
        raise ValueError(f"RTX_VOCAB_PRUNE_MASS={VOCAB_PRUNE_MASS}: use 0 or one of {sorted(KEPT_IDS_BY_MASS)}")
    # The width floor comes from the census at 16,384-token microbatches.
    assert MICROBATCH_TOKENS == RANKED_MICROBATCH_TOKENS, "ranked pruning is sized for 16384-token microbatches"
if VOCAB_PRUNE_TAIL:
    assert VOCAB_PRUNE_MASS, "RTX_VOCAB_PRUNE_TAIL needs RTX_VOCAB_PRUNE_MASS > 0"
    assert VOCAB_PRUNE_TAIL % 128 == 0 and 128 <= VOCAB_PRUNE_TAIL <= 4096, "tail: a multiple of 128, 128-4096"
assert os.environ.get("RTX_VOCAB_PRUNE_EXACT_END", "0") in ("0", "1"), "RTX_VOCAB_PRUNE_EXACT_END: 0 or 1"
if VOCAB_PRUNE_EXACT_END:
    assert VOCAB_PRUNE_MASS, "RTX_VOCAB_PRUNE_EXACT_END needs RTX_VOCAB_PRUNE_MASS > 0"
if VOCAB_PRUNE_FROM_STEP or VOCAB_PRUNE_UNTIL_STEP:
    assert VOCAB_PRUNE_MASS, "the stage gate needs RTX_VOCAB_PRUNE_MASS > 0"
    assert VOCAB_PRUNE_FROM_STEP >= 0 and (not VOCAB_PRUNE_UNTIL_STEP or VOCAB_PRUNE_UNTIL_STEP > VOCAB_PRUNE_FROM_STEP), \
        "stage gate: 0 <= from < until"
if VOCAB_PRUNE_TAIL_POWER != 1:
    assert VOCAB_PRUNE_TAIL, "RTX_VOCAB_PRUNE_TAIL_POWER needs a tail (stratified mode)"
    assert 0 <= VOCAB_PRUNE_TAIL_POWER <= 1, "RTX_VOCAB_PRUNE_TAIL_POWER: 0 to 1"
if VOCAB_PRUNE_SWITCH:
    assert VOCAB_PRUNE_MASS and VOCAB_PRUNE_TAIL, "RTX_VOCAB_PRUNE_SWITCH needs a mass and a tail (stratified mode)"
    _starts = [start for start, _, _ in VOCAB_PRUNE_SWITCH]
    assert _starts == sorted(set(_starts)) and _starts[0] > VOCAB_PRUNE_FROM_STEP, \
        "switch steps: increasing, after the gate's start"
    for _, _mass, _tail in VOCAB_PRUNE_SWITCH:
        assert _mass in KEPT_IDS_BY_MASS, f"switch mass {_mass}: one of {sorted(KEPT_IDS_BY_MASS)}"
        assert _tail % 128 == 0 and 128 <= _tail <= 4096, "switch tail: a multiple of 128, 128-4096"

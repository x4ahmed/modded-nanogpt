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

assert MICROBATCH_TOKENS % 2048 == 0, "microbatches are whole 2048-token units"
assert NGRAM_DIVISOR >= 1 and NGRAM_DIVISOR & (NGRAM_DIVISOR - 1) == 0 and NGRAM_DIVISOR <= 2048
assert VAL_SEQ_LEN % 128 == 0, "validation forwards are whole attention blocks"
assert EVAL_CE_SLAB_ROWS > 0 and WARMUP_MICROBATCHES >= 1
assert MAX_STEPS is None or MAX_STEPS >= 1

# NaN bisection only (RTX_NAN_PROBE, portable/nan_probe.py): drop fp8 from one half of the training
# forward and keep it in the other, to tell an fp8 attention projection NaN from an fp8 MLP one. The
# bf16 halves they fall back to are the validation path, which has no backward, so these are legal in
# a forward-only probe and nowhere else. A real run leaves both on.
FP8_ATTN = os.environ.get("RTX_NO_FP8_ATTN") != "1"
FP8_MLP = os.environ.get("RTX_NO_FP8_MLP") != "1"

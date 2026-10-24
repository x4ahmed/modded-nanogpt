# Sampled-softmax schedule for one GPU (awaiting GPU validation)

Branch `experiment/sampled-softmax-schedule`, from `baseline` (`5be1ba0`). `RTX_SAMPLED_SCHEDULE`
selects the candidate counts in `track_1_short/sampled_softmax.py`; the default `record` is the baseline.

## Idea

Early in training the loss is a softmax over P candidate classes, not all 50,304: every class that is a
target in the set's tokens, plus negatives up to P. The record builds one set per rank-step, from
16,384 / 32,768 / 49,152 tokens in stages 0 / 1 / 2. The port builds one per 16,384-token microbatch, so
from stage 1 on its sets contain fewer distinct targets at the same P, i.e. more negatives than the
record uses, and the lm_head GEMMs and CE kernel do more work than the record's schedule intended.

Distinct targets on the first training shard (3,000 windows each, measured 2026-10-25):

| Tokens per set | Mean | p99 | Max |
| --- | --- | --- | --- |
| 16,384 | 4,278 | 4,881 | 5,197 |
| 32,768 | 6,993 | 7,761 | 8,137 |
| 49,152 | 9,150 | 10,004 | 10,419 |

## Presets

| Stage (steps) | `record` | `record_late` | `matched` | `matched_late` |
| --- | --- | --- | --- | --- |
| 0 | 10,240 | 10,240 | 10,240 | 10,240 |
| 1 | 10,240 | 10,240 | 8,192 | 8,192 |
| 2, first two thirds | 14,336 | 14,336 | 10,240 | 10,240 |
| 2, last third | 24,576 | 24,576 | 20,480 | 20,480 |
| 3, batch taper (~67 steps) | full | 24,576 | full | 20,480 |
| 4, extension (20 steps) | full | full | full | full |

**Run `record_late` only.** Against `record` it changes nothing but the batch taper, so its result is a
claim about the record's schedule that could transfer to 8xH100. `matched` / `matched_late` only correct
this port's per-microbatch sets (the record's per-rank sets are already matched on 8 GPUs), so they would
improve the local port only and are kept but not run.

`matched` keeps the record's number of negatives per set (record P minus the record's distinct targets),
added to this port's 4,278 distinct targets and rounded up to a multiple of 2,048 (the CE kernel width).
`matched_late` also samples the batch taper, testing the record's note that the count must reach full
resolution before the taper. The smallest count, 8,192, is far above the largest microbatch seen (5,197
distinct targets). The matched presets require 16,384-token microbatches (config refuses others).

## Expectations

The training math is unchanged except the candidate sets, so loss should move little; time should drop.
Rough FLOP estimate: `matched` about 2% of training time, `matched_late` about 3-4%. Judge by the
screening rule: one paired seed; loss within ±0.002 of the baseline seed and lower time counts as a win,
then confirm with five seeds.

## Commands

GPU smoke (compiles every count of the preset during warmup): `python -u tests/check_rtx5090.py
--sampled-schedule record_late --seed 999`. Full run: `RTX_SAMPLED_SCHEDULE=record_late`; W&B name
`sampled-record-late-1gpu-seed-<seed>`. Re-run baseline seed 999 (`record`) in the same allocation
for a same-session time comparison. Rough expected saving: about 1-2% of training time.

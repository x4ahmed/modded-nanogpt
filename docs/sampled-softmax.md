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

| Stage (steps) | `record` | `matched` | `matched_late` |
| --- | --- | --- | --- |
| 0 | 10,240 | 10,240 | 10,240 |
| 1 | 10,240 | 8,192 | 8,192 |
| 2, first two thirds | 14,336 | 10,240 | 10,240 |
| 2, last third | 24,576 | 20,480 | 20,480 |
| 3, batch taper (~67 steps) | full | full | 20,480 |
| 4, extension (20 steps) | full | full | full |

`matched` keeps the record's number of negatives per set (record P minus the record's distinct targets),
added to this port's 4,278 distinct targets and rounded up to a multiple of 2,048 (the CE kernel width).
`matched_late` also samples the batch taper, testing the record's note that the count must reach full
resolution before the taper. The smallest count, 8,192, is far above the largest microbatch seen (5,197
distinct targets). Both presets require 16,384-token microbatches (config refuses others).

## Expectations

The training math is unchanged except the candidate sets, so loss should move little; time should drop.
Rough FLOP estimate: `matched` about 2% of training time, `matched_late` about 3-4%. Judge by the
screening rule: one paired seed; loss within ±0.002 of the baseline seed and lower time counts as a win,
then confirm with five seeds.

## Commands

GPU smoke (compiles every new count during warmup): `python -u tests/check_rtx5090.py --sampled-schedule
matched --seed 999`. Full run: `RTX_SAMPLED_SCHEDULE=matched` (or `matched_late`); W&B names
`sampled-matched-1gpu-seed-<seed>` / `sampled-matched-late-1gpu-seed-<seed>`.

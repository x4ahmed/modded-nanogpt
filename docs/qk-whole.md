# Whole-matrix Q/K whitening (optimizer only)

Branch `experiment/qk-whole`, from `baseline` (5be1ba0). Only ANVIL's whitening of `qk_bank` changes; the
model, data, schedule, momentum, lane equalizer, weight decay and the V/O and MLP banks are untouched.

## What changes

ANVIL keeps `qk_bank` as head-pair groups (256 x 768: two heads of Q, or two of K) and whitens each group on
its own, while V and O are whitened as whole 768 x 768 matrices. Here each layer's three Q groups are
stacked and whitened as one 768 x 768 matrix, and the same for K (`optim/anvil.py`, `anvil_cascade`
`merge_groups=3`; the stacking is a plain reshape, since a layer's groups are consecutive). Momentum, the
lane equalizer and decay still run per head-pair group.

Whitening a whole Q also makes every head pair's update rows orthogonal to the other pairs' rows: head
pairs can no longer all move along the same direction. The per-head V/O experiment
(`experiment/vo-heads`) went the other way, from whole matrices to heads, and lost 0.0124 loss: the
cross-head constraint helped there. This tests whether it helps Q and K too.

- `model/gpt.py` marks the bank (`qk_bank.whiten_merge = 3`); `AnvilBank.whiten_merge` carries it to the
  cascade. A rank's slice must hold whole layers: 8 GPUs hold 6 groups (one layer's Q and K) each.
- Narrow layers keep their unused head rows at zero.
- Cost: a 768 x 768 whitening per Q/K instead of three 256 x 768 ones, a few hundredths of a second on 8xH100.
- The run log prints `QK_WHITENING whole_matrix groups_per_matrix=3`; W&B runs are `qk-whole-1gpu-seed-<seed>`.

## Checks (CPU, `tests/test_qk_whole.py`)

The merged update equals running the unchanged whitening on each whole Q/K alone, bit for bit; each whole
matrix's singular values are ~1; two head-pair groups with near-identical gradients become orthogonal
(per group, they overlap); the momentum state is identical either way; unused rows stay zero. Banks
without the mark are bitwise identical to the baseline branch's cascade. On a full-size layer, the merged
update has the same norm as the per-group one but differs by ~48%: a real change, either sign possible.

## Screen

Seed 999, full schedule. Continue to five seeds only if the loss is below the baseline's seed-999 runs
(3.294980, 3.295181, 3.295634) by more than 0.002, which the record prices at ~0.33 s of 8xH100 time.

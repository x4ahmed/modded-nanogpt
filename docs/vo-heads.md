# Per-head V/O whitening (optimizer only)

Branch `experiment/vo-heads`, from `baseline` (5be1ba0). Only ANVIL's whitening of `vo_bank` changes; the
model, data, schedule, momentum, lane equalizer, weight decay and the Q/K and MLP banks are untouched.

## What changes

ANVIL whitens Q and K in groups of two heads (`qk_bank`, 256 x 768 blocks) but V and O as whole 768 x 768
matrices. Whitening a whole V drives all its singular values to 1 together, which also makes every head's
update rows orthogonal to every other head's: heads are forced to update in different directions.

Here each head's V rows and O columns are whitened as their own matrix (`optim/anvil.py`
`vo_head_blocks`: 6 blocks of 128 x 768 per matrix, O blocks transposed so all blocks share one shape).
Each head still gets a full-strength update (all its singular values ~1, the same total norm), but heads
with similar gradients can now update along similar directions, as Q and K heads already can.

- `model/gpt.py` marks the bank (`vo_bank.whiten_head_dim = 128`); `AnvilBank.whiten_head_dim` carries it
  to `anvil_cascade(head_dim=...)`. A rank's slice must hold whole V/O pairs (8 GPUs: 2 matrices each).
- Half-width layers 1 and 8 keep their zero rows/columns; each head's live half is whitened.
- Cost: per-head blocks are several times cheaper to whiten (128 x 128 Grams instead of 768 x 768).
- The run log prints `VO_WHITENING per_head head_dim=128`; W&B runs are `vo-heads-1gpu-seed-<seed>`.

## Checks (CPU, `tests/test_vo_heads.py`)

The per-head update equals running the unchanged whitening on each head block alone, bit for bit. Each
head block's singular values are ~1, and two heads with near-identical gradients keep overlapping
updates (whole-matrix whitening makes them orthogonal). The momentum state is identical either way. The
Q/K-, V/O- and MLP-shaped banks without the mark are bitwise identical to the baseline branch's cascade.

## Why it could transfer, and the screen

Validation loss does not depend on the hardware, and the optimizer math here is the record's. The
record prices val at ~164 ms of 8xH100 time per millinat (0.001 loss; ANVIL2 README ablation), so a
0.002 gain is worth ~0.33 s of its 39.9 s run.

Screen: seed 999, full schedule. Continue to five seeds only if the loss is below the baseline's seed-999
runs (3.294980, 3.295181) by more than 0.002. The direction is not predictable: the record keeps head
pairs for Q/K, but nothing in it tests V/O per head.

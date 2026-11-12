# Vocabulary pruning (Phase 2, idea 2)

Branch: `experiment/vocab-prune`, directly from frozen `baseline` (`5be1ba0`). Idea (Notion, 2026-11-05):
check how often each token appears in the training data and drop the tokens that are not needed.
Status: ranked sweep run 2026-11-11: every mass wrecks the loss (results at the end of "Ranked pruning
sweep"). Its fix, stratified pruning (last section), ran at m = 0.20 / 0.15 / 0.10 (t = 1,024): loss
+0.024 to +0.039, time -5.4% to -8.0%; none meets the 0.002 bar. Exact pruning (mass 0) was never run.

## Where the vocabulary still costs time

The record already prunes the vocabulary twice:

- Sampled softmax (record #360, `track_1_short/sampled_softmax.py`): steps 0-1106 normalize the training
  loss over P = 10,240 / 14,336 / 24,576 candidates (every target of the microbatch plus uniform stride
  negatives), not over 50,304 ids. The width is P, so a smaller vocabulary does not change these steps.
- Canonical mask (record #350, `track_1_short/canonical_mask.py`): validation drops ids the GPT-2
  tokenizer can never emit after the previous token.

| Vocabulary-sized work | When | Gain from a smaller vocabulary |
| --- | --- | --- |
| lm_head GEMMs and CE, full softmax | steps 1107-1193 (87 steps, 24.6M of 328.7M training tokens, 7.5%) | proportional to the ids removed |
| lm_head GEMMs and CE, sampled | steps 0-1106 | none (width P) |
| embed / value-embedding lookups | every step | none (gathers) |
| Adam on lm_head, embed, 4 value-embedding planes (6 x V x 768) | update steps | small (memory-bound) |
| validation forward | after training | none: off the clock |

Size of the remaining lever, from our own measurement: `record_late` (2026-10-28, same allocation as its
control) trained the 67 taper steps at P = 24,576 instead of 50,304 and saved 1.7 s, about 1 ms per
1,000 output ids per batch-20 step. Over the 87 full-softmax steps that is ~0.1 s per 1,000 ids removed;
a FLOP bound (24.6M tokens x 768 x 1,000 ids x 6) gives at most ~0.4 s. Same-allocation pair noise in
time is ~1-2 s, so pruning must remove several thousand ids before a pair can see it.

Out of scope: retokenizing with a smaller BPE (changes the token stream and the loss-per-GPT-2-token
metric) and choosing ids from validation counts (peeking). Any id removed from the output must keep a
finite probability: under the -60 mask a validation token on it costs ~80 nats.

## Step 1: census

`tools/token_census.py` replays exactly the token stream the one-GPU run trains on (the loader's
BOS-aligned slicing, per-stage document cuts, 16,384-token microbatches, 1194 steps, 328,663,040 target
tokens) and counts every id as a target and as an input; it also counts the 10,485,760 validation
targets, only to measure the risk of a train-derived choice. CPU only, no training, no GPU needed.

```bash
source ~/venvs/nanogpt/bin/activate
cd ~/speed-run/modded-nanogpt
git fetch origin
git switch experiment/vocab-prune
git pull --ff-only origin experiment/vocab-prune
python -u tools/token_census.py
```

It prints a short report (ids that occur, ids needed for 99-99.999% coverage, and for train counts
below 1 / 10 / 100 / 1000: how many ids, their share of training targets, and how many validation
tokens fall on them), then `CENSUS_COMPLETE dir=logs/token-census/<UTC>`. The folder keeps
`counts.npz` (per-id counts) and `summary.json`. CPU tests: `tests/test_token_census.py` (slicing
identical to the record's `Shard` on synthetic shards, counts against brute force, schedule totals).

## Census result (2026-11-10)

Cluster folder `logs/token-census/20261110T214058Z`: 328,663,040 training targets, 1194 steps, 5 shards.

| Ids dropped (train count) | Ids | Kept | Share of training targets | Validation tokens on them | Time saved (anchor - FLOP bound) |
| --- | --- | --- | --- | --- | --- |
| 0 (never seen) | 268 | 49,989 | 0 | 1 (` oun`, id 16103) | 0.02 - 0.10 s |
| < 10 | 587 | 49,670 | 4.5e-6 | 68 (43 ids) | 0.04 - 0.22 s |
| < 100 | 2,337 | 47,920 | 2.9e-4 | 3,580 (1,188 ids) | 0.17 - 0.88 s |
| < 1000 | 26,841 | 23,416 | 3.8% | 421,761 (25,537 ids) | 1.9 - 9.7 s, only as a two-level softmax |

36,968 ids cover 99% of the training targets, 46,223 cover 99.9%, 48,696 99.99%, 49,530 99.999%;
49,510 ids occur even within the 24.6M tokens of the full-softmax steps. The 268 never-seen ids (the
same 268 never occur as inputs) are mostly byte tokens that cannot occur in valid UTF-8 (0xC0, 0xC1,
0xF5-0xFF) and control characters (0x00-0x0C), plus a few such as `{` (id 90).

- Exact pruning removes 315 never-target columns (with the 47 padding ids): the full-softmax width
  drops from 50,304 to 50,048 at 128-column alignment (through the sampled path's 2048 alignment it
  would grow to 51,200). At most ~0.1 s, far below pair noise.
- Dropping seen-but-rare ids needs a fallback probability: hard-masking the < 10 set alone would add
  ~5e-4 loss through its 68 validation tokens. Up to count < 100 it still saves under 1 s.
- Only a two-level output (a frequent head of ~23.4K ids plus a rare-token cluster scored on the 3.8%
  of tokens that need it) cuts enough width: 1.9 s (measured anchor) to 9.7 s (FLOP bound), 0.2-1.1%.
  It replaces the flat softmax for the whole run (the sampled steps train no rare cluster), needs new
  fused fp8 CE kernels and canonical-mask handling, and usually costs some loss, which extra steps
  would have to buy back.
- Option B below (eval-time mask of the 268 ids) would remove only floor-level mass and put one
  validation token at ~80 nats.

No measurable time lever was found; no pair has been run. Closing or continuing is the user's call.

## Dynamic pruning run (verifies the prediction)

User request (2026-11-10): implement pruning in the run and measure it. `track_1_short/portable/vocab_prune.py`,
wired into `portable/sampled.py` (one GPU only; the 8-GPU path is untouched):

- Every microbatch marks its targets as seen on the GPU (one `index_fill_`), from step 0. No offline
  counts are used: the set is the run's own targets so far.
- The record's full-softmax steps 1107-1193 instead run the sampled-softmax kernels over a fixed-width
  set C of PRUNED_WIDTH = 50,048 ids: every id seen as a target so far, filled with never-seen ids
  (lowest first, so the 47 padding ids and the highest never-seen ids are left out). Every target of
  the microbatch is in C, so its loss term is exact; only never-seen ids leave the normalizer and get
  no lm_head gradient on those steps. If a target outside C appears, C is rebuilt around it (one sync).
  If the seen ids ever exceed 50,048, that microbatch uses the full softmax (logged; not expected:
  the census found 49,989 for the whole run).
- 50,048 is the largest exact cut (the next 128-column step, 49,920, is below 49,989). The CE kernel is
  compiled at that width; like 50,304 it relies on the kernel's tail guard (P % 2048 != 0).
- Per step: one lm_head row gather and transpose (the fp8 cache changes only between steps); per
  microbatch: a host membership check and a device lookup of target positions. The sampled path also
  densifies the lm_head gradient (a 77 MB pass) that the full path does not need.
- Validation is unchanged (all 50,304 ids, canonical mask). Logs: `VOCAB_PRUNE width=...` at start,
  `VOCAB_PRUNE warmup ...` after warmup, `VOCAB_PRUNE run builds=... seen_total=... left_out=...`
  before `final_val_loss`.

Prediction: time change within about -0.1 to +0.2 s (256 fewer columns on 87 steps, minus the gather
and densify overheads) and loss change within same-seed noise (~0.001). One pair cannot resolve
effects below its ~1-2 s time noise, so "no detectable change" is the expected, confirming outcome.

```bash
source ~/venvs/nanogpt/bin/activate
cd ~/speed-run/modded-nanogpt
git fetch origin
git switch experiment/vocab-prune
git pull --ff-only origin experiment/vocab-prune
git show-ref --verify --quiet refs/heads/baseline || git branch baseline origin/baseline
python -u tools/run_pair.py --seeds 999
```

`tools/run_pair.py` is copied unchanged from `experiment/attnres-last-init` (GPU smoke gate, which
also compiles and runs the pruned steps during warmup; then baseline, then candidate; W&B names
`baseline-for-vocab-prune-1gpu-seed-999` and `vocab-prune-1gpu-seed-999`). CPU tests:
`tests/test_vocab_prune.py` (set exactness, rebuilds, fallback, warmup interleaving, schedule).

## Ranked pruning sweep (all steps)

User plan (2026-11-10). Knob `RTX_VOCAB_PRUNE_MASS=m`: the share of training text whose tokens may
leave the softmax. On every step the candidate set is the microbatch's targets (exact loss, no fallback
needed) plus the most frequent ids seen so far in this run, at width min(record width, W(m)), where W(m)
= the ids covering 1 - m of the training text (census, 2026-11-10 rerun), rounded up to 128, at least
6,144 (the census max is 5,292 distinct targets per 16,384-token microbatch; the loader is deterministic,
so this holds for every run). Built on the GPU (`RankedCandidates`: per-step argsort of the run's target
counts, per-microbatch selection; no host builder, no upload, no sync). Validation untouched.

| m | Widths used (record capped at W) | Output-layer time saved: measured rate - FLOP bound |
| --- | --- | --- |
| 0.001 | 10,240 / 14,336 / 24,576 / 46,336 | 0.3 - 1.5 s |
| 0.01 | 10,240 / 14,336 / 24,576 / 36,992 | 1.0 - 5.0 s |
| 0.05 | 10,240 / 14,336 / 20,096 | 3.0 - 15.3 s |
| 0.10 | 10,240 / 11,136 | 6.2 - 31.9 s |
| 0.15 | 6,656 | 10.3 - 52.7 s |
| 0.20, 0.30 | 6,144 (floor; identical runs) | 10.8 - 55.3 s |

The whole output layer costs 16.8 s (measured rate) to 86.4 s (FLOP bound) over the run. Even where the
width equals the record's, the fill changes from uniform stride negatives to frequency-ranked ids (and
the host builder to the device one), so low-m points also measure that change. Main risk: ids below the
cut are never negatives unless they are targets, so their logits may drift up and cost validation loss;
the sweep measures where that starts. Part of any gain may be port-specific: the record's 8-GPU
microbatches are 3x larger, so its candidate sets need more room.

Each point is its own pair (`--candidate-env` sets the knob for the gate and candidate only):

```bash
source ~/venvs/nanogpt/bin/activate
cd ~/speed-run/modded-nanogpt
git fetch origin
git switch experiment/vocab-prune
git pull --ff-only origin experiment/vocab-prune
git show-ref --verify --quiet refs/heads/baseline || git branch baseline origin/baseline
for m in 0.15 0.10 0.05 0.01; do
  python -u tools/run_pair.py --seeds 999 --label vocab-prune-m$m --candidate-env RTX_VOCAB_PRUNE_MASS=$m
done
```

Logs: `VOCAB_PRUNE mode=ranked mass=... width=... (widths by stage)` at start and
`VOCAB_PRUNE run overflow_microbatches=0 distinct_seen=...` before `final_val_loss`. CPU tests:
`tests/test_vocab_prune.py` (widths, ranking and selection, target/prefix mapping, config validation).

### Sweep results (2026-11-11, seed 999, commit 74cf685, one pair per mass)

| m | Baseline loss / s | Candidate loss / s | Loss delta | Time delta |
| --- | --- | --- | --- | --- |
| 0.15 | 3.294583 / 863.586 | 3.883074 / 790.490 | +0.5885 | -73.1 s (-8.46%) |
| 0.10 | 3.295893 / 863.220 | 3.893151 / 816.955 | +0.5973 | -46.3 s (-5.36%) |
| 0.05 | 3.295336 / 865.803 | 3.845152 / 836.872 | +0.5498 | -28.9 s (-3.34%) |
| 0.01 | 3.295875 / 862.363 | 3.653614 / 847.660 | +0.3577 | -14.7 s (-1.70%) |

All four runs: 1194 updates, no overflow (every target was a candidate), 49,989 ids seen. The four
baselines agree (3.2946-3.2959, 862.4-865.8 s).

- Loss: 180-300x the 0.002 bar, at every mass. Even m = 0.01, whose widths equal the record's on steps
  0-1106, loses 0.36, so the cause is the fill, not the width. With frequency-ranked fill, an id outside
  the set is a negative only when it happens to be a target elsewhere in the microbatch; tens of
  thousands of ids are almost never pushed down, their logits stay near their high initial softcap value
  (~15 of 23), and at validation, which scores all 50,304 ids, they absorb probability. The record's
  uniform stride negatives exist to visit every class (sampled_softmax.py); this sweep shows why.
- Time: real but partly not from the width. m = 0.01 saved 14.7 s against a 1-5 s width ceiling, and
  m = 0.15 saved 73 s against a 10-53 s ceiling: about 10-20 s comes from building candidate sets on the
  GPU (no per-microbatch host build and synchronous uploads), which the port does on the CPU.
- Follow-ups this suggests, not run: (1) the device-side builder with the record's uniform stride
  negatives (expected: same loss as the baseline, ~1% faster; a port-only gain, since the 8-GPU record
  already overlaps its host build); (2) ranked fill mixed with uniform negatives, so every id is still
  visited. Continuing or closing is the user's call.

## Stratified pruning: the fix (2026-11-11; results at the end)

User request: keep the time saving, keep the loss, and only in a form that transfers to the 8-GPU
record (no port-only gains). `RTX_VOCAB_PRUNE_TAIL=t` together with a mass m:

- Exact part: the microbatch's targets plus the best-ranked ids, at width ranked_width(m) (as in the
  ranked sweep). Every target stays exact.
- Tail: t ids drawn uniformly at random from ALL remaining ids (own torch.Generator, seed 20011, reset
  after warmup; the global RNG streams are untouched). Each carries weight w = (remaining ids) / t.
- Loss: normalizes over Z_hat = sum_exact exp(z) + w * sum_tail exp(z), an unbiased estimate of the full
  softmax's Z (CPU test: mean within 2% over 400 draws; without weights the same sets miss over half the
  mass). Every id is drawn now and then, so every id is still pushed down with the right average
  strength: this removes the ranked sweep's failure. log Z_hat is slightly biased low (Jensen); the
  tail size sets that noise.
- Kernel: `portable/weighted_ce.py` compiles the record's CE kernel source with four text substitutions
  that add col_logw[c] = log w_c to every exp() (forward sum, dense gradient, target correction); a CPU
  test checks that removing them gives back the record's source exactly. Record kernels are untouched.
  First GPU check is the pair's smoke gate.
- Prefix targets (stages 0-1) are kept only on exact columns, as the record drops prefix targets outside
  its set.
- Widths: min(record width, ranked_width(m) + t). With t = 1,024:

| m | Widths | Width-based time ceiling (measured rate - FLOP bound) |
| --- | --- | --- |
| 0.20 (and above: floor) | 7,168 | 9.8 - 50.1 s |
| 0.15 | 7,680 | 9.3 - 47.5 s |
| 0.10 | 10,240 / 12,160 | 5.6 - 28.9 s |

Transfer: the estimator is the transferable part. On 8xH100 each rank's microbatch has ~11k distinct
targets (sampled_softmax.py), so the record's floor is ~12-13k plus the tail; there it would let the
14,336 / 24,576 stages and the 87 full-softmax steps narrow. The local gain from building sets on the
GPU is port-only and is not claimed.

```bash
source ~/venvs/nanogpt/bin/activate
cd ~/speed-run/modded-nanogpt
git fetch origin
git switch experiment/vocab-prune
git pull --ff-only origin experiment/vocab-prune
git show-ref --verify --quiet refs/heads/baseline || git branch baseline origin/baseline
for m in 0.20 0.15 0.10; do
  python -u tools/run_pair.py --seeds 999 --label vocab-prune-m$m-t1024 \
    --candidate-env RTX_VOCAB_PRUNE_MASS=$m --candidate-env RTX_VOCAB_PRUNE_TAIL=1024
done
```

Logs: `VOCAB_PRUNE mode=stratified mass=... tail=1024 width=...` at start; `VOCAB_PRUNE run
overflow_microbatches=0 ...` before `final_val_loss`.

### Stratified results (seed 999, commit 2bf3d65)

| m, t | Baseline loss / s | Candidate loss / s | Loss delta | Time delta |
| --- | --- | --- | --- | --- |
| 0.20, 1024 (width 7,168 on every step) | 3.294050 / 862.441 | 3.332638 / 793.914 | +0.0386 | -68.5 s (-7.95%) |
| 0.15, 1024 (width 7,680 on every step) | 3.295977 / 843.1 | 3.329950 / 785.8 | +0.0340 | -57.3 s (-6.79%) |
| 0.10, 1024 (widths 10,240 / 12,160) | 3.294737 / 850.9 | 3.318744 / 805.0 | +0.0240 | -45.9 s (-5.40%) |

No overflow; 49,989 ids seen. Against the ranked sweep (+0.59 at m = 0.15) the reweighted tail recovers
~94% of the lost loss at almost the same speed, but every point is 12-19x the 0.002 bar. The trade is
nearly linear: ~0.0044-0.0050 loss per 1% time at all three masses, so no mass is a sweet spot.
m = 0.10 keeps the record's width in stage 1 and still loses +0.024, so narrowing alone does not explain
the damage. Suspects: the last 87 steps are no longer exact (record_late: even P = 24,576 there cost
+0.0064), and the estimator's noise and low bias of log Z (1,024 draws at weight ~42).
Candidate next tests: an exact ending (full softmax on steps 1107-1193 in stratified mode), or a larger
tail (2,048). The user chose the exact ending (2026-11-11).

## Exact ending (2026-11-11; result at the end)

`RTX_VOCAB_PRUNE_EXACT_END=1` (needs a mass): the record's full-softmax steps 1107-1193 (87 steps, 24.6M
tokens) keep the record's full softmax (`prepare` returns None, the record's own path); the sampled steps
stay stratified. Tests whether the inexact ending causes the loss gap. Estimated cost against plain
m = 0.20: 43,136 extra columns x 24.6M tokens = 3-16 s, so roughly -6.1% to -7.6% instead of -7.95%.

```bash
source ~/venvs/nanogpt/bin/activate
cd ~/speed-run/modded-nanogpt
git fetch origin
git switch experiment/vocab-prune
git pull --ff-only origin experiment/vocab-prune
python -u tools/run_pair.py --seeds 999 --label vocab-prune-m0.20-t1024-xe \
  --candidate-env RTX_VOCAB_PRUNE_MASS=0.20 --candidate-env RTX_VOCAB_PRUNE_TAIL=1024 \
  --candidate-env RTX_VOCAB_PRUNE_EXACT_END=1
```

Log: `VOCAB_PRUNE mode=stratified ... exact_end=1: full softmax on steps 1107-1193 (87 steps)`.

### Exact-ending result (seed 999, commit 2411e4a)

| Run | Baseline loss / s | Candidate loss / s | Loss delta | Time delta |
| --- | --- | --- | --- | --- |
| m 0.20, t 1024, exact end | 3.295625 / 847.191 | 3.319397 / 794.359 | +0.0238 | -52.8 s (-6.24%) |
| m 0.20, t 2048, exact end (2026-11-12) | 3.295598 / 833.623 | 3.304879 / 795.297 | +0.0093 | -38.3 s (-4.60%) |

Tail 2,048 (width 8,192) cuts the gap by 61% (+0.0238 -> +0.0093): sampling noise on steps 0-1106 is the
main cause. The candidate itself costs only ~1 s more (795.3 vs 794.4 s); most of the smaller % saving is
this pair's faster baseline (833.6 vs 847.2 s, session drift). ~0.0020 loss per 1% time; still 4.6x the
0.002 bar. If the gap scales ~1/t, t = 4,096 (width 10,240 = the record's stage-1 width) lands near
+0.005; stages 2-4 (14,336 / 24,576) are then the only narrowing.

### Tail only, t = 4,096 (2026-11-12; isolates the tail, compare with t = 1,024 at m = 0.20)

| Run | Baseline loss / s | Candidate loss / s | Loss delta | Time delta |
| --- | --- | --- | --- | --- |
| m 0.20, t 4096, no exact end (width 10,240) | 3.294255 / 833.459 | 3.306522 / 841.328 | +0.0123 | +7.9 s (+0.94%) |

- Loss: the tail alone cuts the gap from +0.0386 (t = 1,024) to +0.0123 (-68%). Exact end + t = 2,048
  (+0.0093) is still better, so both parts help.
- Time: NOT a builder overhead. Per-window step times (ms/step, from the console logs) show the t = 4,096
  candidate as fast as or faster than the t = 1,024 candidate in 11 of 12 windows; only steps 900-1000
  jump to 1,489.6 ms/step (neighbours ~950), ~54 s in one window, which is the whole regression. Its own
  baseline shows no jump there. The log narrows it to steps 926-950: 76.4 s for 25 steps (~24 s
  normally), a ~52 s stall with no warning lines. Cause unknown: node interference or a one-off stall. Without that window
  the candidate is ~787 s, about -5.5% against its baseline (estimate, not a measurement).

Against plain m = 0.20 (+0.0386 / -7.95%), the exact ending removes 0.0148 of the loss gap (38%) for
1.7% of time (~15 s, the high end of the 3-16 s estimate). It matches m = 0.10's loss (+0.0240) while
saving more time (-6.24% vs -5.40%): ~0.0038 loss per 1% time, against ~0.005 before. Still 12x the
0.002 bar. The remaining +0.024 comes from the sampled steps 0-1106 (estimator noise / low log Z bias,
narrow width).

## Options considered before the census

- Fewer than ~5,000 removable ids (train count 0, validation hits ~0): the time gain is below pair noise.
  This is the measured case (268 ids, one validation hit).
- More: two candidates, each its own paired run (`tools/run_pair.py` from `experiment/attnres-last-init`):
  - A. Pruned output in the full-softmax steps: run steps 1107-1193 through the sampled-softmax path
    with a fixed candidate set = ids seen as training targets so far (every target is in it by
    construction); negatives only from seen ids. Validation keeps all 50,304 ids (metric unchanged).
  - B. Eval-time mask of never-seen ids, an extension of the canonical mask, counted on the fly from
    the run's own targets. No training cost; measured exactly inside one run by computing the
    validation loss with and without the extra mask from the same forward. Expected gain is tiny:
    those ids already sit near the softcap floor.

# Block AttnRes with a last-source init (Phase 2)

Branch: `experiment/attnres-last-init`, directly from frozen `baseline` (`5be1ba0`).
Question (supervisor discussion, 2026-11-05): did Experiment 1 (block AttnRes, five seeds:
+0.036 loss, +10% time) lose because of how it was initialized?

## What is tested

The block AttnRes of Experiment 1 is ported unchanged (commit `7a47c05` cherry-picked; block size 6,
injections on; architecture in `docs/attention-residuals.md`). Its unused static mode is removed. This
branch always trains the candidate; the control is the frozen baseline commit, never a mode of this code.

The only change: each of the 18 depth mixes (17 sublayer sites plus the final readout) learns one scalar,
`attnres_last_source_bias[site]`, added to the logit of its newest source (the open block, or else the
newest completed block). With n sources the newest logit gains `bias + log(n - 1)`, so at zero query
the newest source has weight `sigmoid(bias)` and the other n - 1 share the rest equally.

| | Experiment 1 (zero init) | This branch |
| --- | --- | --- |
| Queries | zero | zero (unchanged) |
| Initial depth weights | uniform, 1/n each | newest source 0.9, the others 0.1/(n - 1) |
| New parameters | none | 18 fp32 scalars, start at logit(0.9) = ln 9 |

A site with one source (layer 0 attention) ignores its bias. The bias uses replicated Adam with the
`mudd_b2` settings (LR 0.008 x 0.25, betas 0.9/0.99, no weight decay), is created without RNG draws
(baseline initialization stream unchanged), is restored by the warmup reset and is not tail-averaged,
like the queries. Extra work per mix: one scalar add.

Why the newest source and not "the running residual stream": Experiment 1's sources are block outputs,
not running totals. Uniform weights give their mean, which is the plain residual sum divided by n, and
pre-norm removes that scale. Putting most weight on the running stream would therefore start exactly
where Experiment 1 started. Weighting the newest source is the reading that actually changes the start.
Isolation is as partial as in Experiment 1 (see `docs/attention-residuals.md`); attribute results to
"block AttnRes as adapted here, with this init".

## Phase 2 protocol: every run is a pair

Each candidate run is paired with a fresh baseline run: same seed, GPU allocation, environment, data
and validation settings. Loss and time are compared only within that pair. Seed 999 screens; continue
with paired seeds 1000-1003 only if the candidate beats its paired baseline by more than 0.002.

```bash
source ~/venvs/nanogpt/bin/activate
cd ~/speed-run/modded-nanogpt
git fetch origin
git switch experiment/attnres-last-init
git pull --ff-only origin experiment/attnres-last-init
git show-ref --verify --quiet refs/heads/baseline || git branch baseline origin/baseline
python -u tools/run_pair.py --seeds 999
```

`tools/run_pair.py` refuses a dirty checkout or a moved baseline and clears stale RTX/short-run flags.
It runs the GPU smoke check (`tests/check_rtx5090.py`: warmup, reset, two updates, validation, memory
gate, routing gradients and Adam updates of queries and biases), then the baseline, then the candidate.
It writes `logs/pairs/<UTC>/seed-<seed>.json` and prints `PAIR_COMPLETE` with both losses, both times
and the deltas. W&B names: `baseline-1gpu-seed-<seed>` and `block-attnres-last-b6-1gpu-seed-<seed>`.
The runner is generic: copy it unchanged to later Phase 2 branches.

## Zero-init reference pair (Experiment 1 rerun)

Experiment 1's five seeds ran before the pairing rule. Its seed-999 rerun is paired the same way, from
`experiment/attention-residuals` (training code identical to 9655fc3): smoke check, baseline, then
`RTX_ROUTING=block_attnres RTX_ATTNRES_BLOCK_SIZE=6 RTX_ATTNRES_INJECT=1 bash run_rtx.sh`.

## Results

| Pair (seed 999) | Baseline loss / s | Candidate loss / s | Loss delta | Time delta |
| --- | --- | --- | --- | --- |
| Zero init (Experiment 1 rerun) | pending | pending | | |
| Last-source init (this branch) | pending | pending | | |

No GPU result yet. CPU tests cover the bias math, gradients, compile, per-site wiring, registration,
config, launcher names and the pair runner; they are not GPU validation.

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
and the deltas. Neither arm uploads itself; afterwards the runner imports both logs to W&B as
`baseline-for-attnres-last-init-1gpu-seed-<seed>` and `attnres-last-init-1gpu-seed-<seed>` (label =
branch name without `experiment/`, or `--label`). A failed upload keeps local results and prints a retry.
The runner is generic: copy it unchanged to later Phase 2 branches.

## Zero-init reference pair (Experiment 1 rerun)

Experiment 1's five seeds ran before the pairing rule. Its seed-999 rerun is paired the same way, from
`experiment/attention-residuals` (training code identical to 9655fc3): smoke check, baseline, then
`RTX_ROUTING=block_attnres RTX_ATTNRES_BLOCK_SIZE=6 RTX_ATTNRES_INJECT=1 bash run_rtx.sh`, both with
`RTX_WANDB_UPLOAD=0`, then two explicit uploads named `baseline-for-attnres-zero-init-1gpu-seed-999` and
`attnres-zero-init-1gpu-seed-999` (the launchers would reuse the old, ambiguous names).

## Results

| Pair (seed 999) | Baseline loss / s | Candidate loss / s | Loss delta | Time delta |
| --- | --- | --- | --- | --- |
| Zero init (Experiment 1 rerun), 2026-11-08 | 3.295493 / 862.153 | 3.332404 / 952.489 | +0.036911 | +90.336 s (+10.48%) |
| Last-source init (this branch), 2026-11-10 | 3.294893 / 863.494 | 3.327843 / 960.098 | +0.032950 | +96.604 s (+11.19%) |

Zero-init pair (user-supplied, one tujestpolin allocation, smoke check first): baseline log
`logs/run-20261108T214316Z-1910059.console.txt`, candidate `logs/run-20261108T220007Z-1912665.console.txt`
(`ROUTING variant=block_attnres block_size=6 inject=1`). Both 1194/1194 updates, memory PASS (peak
allocated 19.99 GiB each; headroom 5.81 / 6.24 GiB). It reproduces Experiment 1's five-seed regression
(+0.0359, +10.1%): the zero-init AttnRes loss is stable across sessions.

Last-source pair (`logs/pairs/20261110T185526Z/seed-999.json`, candidate 8fdfadd, smoke check passed):
baseline `logs/run-20261110T185526Z-31500.console.txt`, candidate `logs/run-20261110T191212Z-33966.console.txt`
(`ROUTING ... last_init=0.9`). Both 1194/1194, memory PASS (19.99 GiB each), both W&B uploads succeeded.
Verdict: still far worse than baseline (+0.0330 loss, +11.2% time), so STOP; no five-seed expansion.
Its loss gap is ~0.004 smaller than zero init's (+0.0369), a one-seed difference across two pairs, so
the init explains at most a small part of the regression. The ~10% slowdown is the block AttnRes
routing itself (the bias adds one scalar per mix). The remaining gap likely comes from what block mode
removes (MUDD routing, residual scaling, shared late attention input), which no init restores.
CPU tests cover the bias math, gradients, compile, per-site wiring, registration,
config, launcher names and the pair runner; they are not GPU validation.

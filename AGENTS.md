# Agent handoff

## Working rules
- Keep replies short and outcome first. Never omit failed checks or material limits.
- Preserve the RTX adaptation and unrelated user changes. Do not restore old v1 plans.
- User permits commits and pushes. Never force-push or move the frozen baseline.
- Every experiment branches directly from baseline; never stack or merge experiments.
- PHASE 2 RULE (user, 2026-11-08): every experiment runs as a pair with a fresh baseline run: same
  seed, GPU allocation, environment, data and validation settings. Compare loss AND time only within
  that pair (tools/run_pair.py). Historical results alone are never the control.
- Start with seed 999; expand to paired seeds 1000-1003 only if loss beats its paired baseline by > 0.002.
- Never SSH/scp to the cluster. The user executes GPU commands and shares output.
- CPU checks are not GPU validation. No speed/quality claim without full-run evidence.
- Update this file after every change/result/decision. Replace stale text; keep under 100 lines.
  Put protocols in docs/ and tests/, large results in logs/.

## Repository and goal
- Goal: reduce training time at matched validation quality on RTX 5090, with a plausible transfer
  path to 8xH100. Actual multi-GPU validation remains necessary.
- Frozen baseline: 5be1ba0ca09293a007997cdce51c036bca3e89ae (branch baseline).
- Active: experiment/attnres-last-init (Phase 2, idea 1), directly from baseline.
- Entry: run_rtx.sh -> train_gpt_rtx.py; shared architecture: track_1_short/.
- Cluster checkout: ~/speed-run/modded-nanogpt; training venv: ~/venvs/nanogpt.
- Commands require one allocated RTX 5090; no concurrent training on that GPU.
- Local CPU tests: C:/Users/Ahmed/anaconda3/python.exe (torch 2.8, CPU); plain `python` is not on PATH.

## This experiment: block AttnRes with a last-source init
- Ports Experiment 1's block AttnRes unchanged (7a47c05 cherry-picked; block size 6, injections on).
  Static mode removed. docs/attention-residuals.md = architecture; docs/attnres-last-init.md = change.
- The branch always trains the candidate: config rejects any other RTX_ROUTING under PORTABLE=1.
  The non-PORTABLE (8-GPU) path keeps baseline routing; block AttnRes is untested there.
- Only change vs Experiment 1: attnres_last_source_bias, one learned scalar per depth mix (17 sites +
  readout), added to the newest source's logit as bias + log(n-1). At zero query the newest source
  (open block, else newest completed block) starts at weight 0.9; the others share 0.1 equally.
- Why newest source: zero init averages block outputs = plain residual direction (pre-norm removes
  the 1/n scale), so "most mass on the running stream" would not change the start; this does.
- Bias optimizer: replicated Adam with mudd_b2 settings (LR x0.25, betas 0.9/0.99, no WD). Created by
  torch.full (no RNG draws); warmup reset restores it; not tail-averaged. Site 0 (one source) ignores it.
- Log line: ROUTING variant=block_attnres block_size=6 inject=1 last_init=0.9; uploader records it.
- RESULT 2026-11-10 (seed-999 pair, 8fdfadd, smoke pass): baseline 3.294893 / 863.494 s, last-init
  3.327843 / 960.098 s: +0.032950 loss, +11.19% time. Misses the > 0.002 bar for more seeds; continuing is the user's call (not stopped). Only ~0.004
  better than zero init's +0.0369 (one seed, separate pairs): init is not the main cause of the
  AttnRes regression; block mode's removed MUDD/residual pieces likely are. docs/attnres-last-init.md.
- Run long jobs inside tmux on the login node (an earlier attempt died with a frozen SSH terminal).

## Commands
- Pair: python -u tools/run_pair.py --seeds 999 (clean checkout of this branch; full command block in
  docs/attnres-last-init.md). Gate = GPU smoke; then baseline, then candidate, same allocation.
  Writes logs/pairs/<UTC>/seed-N.json; prints PAIR_COMPLETE with both losses, times and deltas.
  Arms train with RTX_WANDB_UPLOAD=0; the runner then uploads both as baseline-for-<label>-1gpu-seed-N
  and <label>-1gpu-seed-N (label = branch minus experiment/, or --label), outside the timers.
- tools/run_pair.py is generic (candidate = current branch): copy it unchanged to later Phase 2 branches.
- Zero-init reference pair DONE 2026-11-08 (seed 999): baseline 3.295493 / 862.153 s, zero-init
  AttnRes 3.332404 / 952.489 s: +0.036911 loss, +10.48% time; reproduces Exp 1. Details in
  docs/attnres-last-init.md. Uploaded under old names; user renamed both in W&B to the pair names. Was: on
  experiment/attention-residuals, smoke check, baseline, then RTX_ROUTING=block_attnres
  RTX_ATTNRES_BLOCK_SIZE=6 RTX_ATTNRES_INJECT=1 bash run_rtx.sh; uploads off, then explicit uploads as
  baseline-for-attnres-zero-init-1gpu-seed-999 / attnres-zero-init-1gpu-seed-999.
- GPU smoke alone: python -u tests/check_rtx5090.py --seed 999 (--attnres-inject 0 for the other arm).
- CPU: python -m unittest discover -s tests -p 'test_*.py' -v.

## Frozen local settings
- One RTX 5090 (SM120, 32 GB); PORTABLE=1.
- RTX_MICROBATCH_TOKENS=16384; RTX_NGRAM_DIVISOR=32 (2,643,840 n-gram rows).
- Stage accumulation counts 8/16/24/20/8; retain existing gradient scaling.
- RTX_VAL_SEQ_LEN=65536; RTX_EVAL_CE_SLAB_ROWS=4096.
- RTX_WARMUP_MICROBATCHES=2; RTX_MIN_HEADROOM_GIB=1.5.
- Full schedule: 1194 optimizer updates. Unset RTX_MAX_STEPS, RTX_PREFLIGHT and
  NUM_SCHEDULED_ITERATIONS. Short tests truncate, never rescale schedules.
- Keep local validation packing fixed; do not claim upstream metric parity.

## Results so far
- Five-seed baseline at 5be1ba0: loss 3.2948572 +/- 0.00104874 SD; time 861.635 +/- 1.608645 s SD.
  Same-seed reruns vary ~0.001 in loss; wall time drifts ~15 s between sessions (hence pairs).
- Phase 1, none promoted. Five seeds, unpaired: block AttnRes (Exp 1, zero init) +0.036 / +10%;
  minimal AttnRes readout +0.010 / +0.6%; bigram prior +0.0025 / +0.85%; Gated DeltaNet +0.0041 / +8.2%.
- Phase 1, seed 999: sampled-softmax taper +0.0064; per-head V/O whitening +0.012; whole Q/K
  whitening -0.001 (noise); layer smear -0.0009 / +4%; low-rank parallel MLP +0.0021 / -0.02% (paired).
- Eval-time sweep (temperature/windows): record settings already best. Async uploads -1.7% (local-only,
  kept on its own branch). Gram NS: little total-run potential.
- Exp 2 (minimal readout) already started its readout last-heavy (x at 0.9), but only the readout.
- Phase 2 ideas (Notion, 2026-11-05): AttnRes init (this branch), mHC, vocabulary pruning.
  Notion "Ideas vs Results" Phase 2 scoreboard: row 1 = zero-init pair, row 2 = last-init pair.
- Canonical-mask warnings and rebuild time remain part of measured training time.

## Logging
- Trainer log: logs/<UUID>.txt; launcher console: logs/run-*.console.txt.
- W&B entity/project i-learn/speedrun, uploaded only after training ends.
- Uploader: ~/venvs/wandb-upload/bin/python (WANDB_UPLOAD_PYTHON override).
- Paired names (Phase 2): baseline-for-<label>-1gpu-seed-N and <label>-1gpu-seed-N, set by the runner.
  Plain baseline-1gpu-seed-N = historical/unpaired baselines only. Standalone runs of this branch
  self-name block-attnres-last-b6-1gpu-seed-N (-noinject suffix). Names are display-only (IDs = log hash).
- Repeated names retain distinct IDs; pair JSON records corresponding console logs.
- Never expose API keys or upload raw source logs. Upload failure retains local results.

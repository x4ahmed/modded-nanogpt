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
- Never SSH/scp to the cluster. The user executes cluster commands and shares output.
- CPU checks are not GPU validation. No speed/quality claim without full-run evidence.
- Whether to continue or close an idea is the user's decision; report results, do not declare stops.
- Update this file after every change/result/decision. Replace stale text; keep under 100 lines.
  Put protocols in docs/ and tests/, large results in logs/.

## Repository and goal
- Goal: reduce training time at matched validation quality on RTX 5090, with a plausible transfer
  path to 8xH100. Actual multi-GPU validation remains necessary.
- Frozen baseline: 5be1ba0ca09293a007997cdce51c036bca3e89ae (branch baseline).
- Active: experiment/vocab-prune (Phase 2, idea 2), directly from baseline. Trains the candidate:
  vocabulary pruning (portable/vocab_prune.py, knob RTX_VOCAB_PRUNE_MASS); tools/token_census.py.
- Entry: run_rtx.sh -> train_gpt_rtx.py; shared architecture: track_1_short/.
- Cluster checkout: ~/speed-run/modded-nanogpt; training venv: ~/venvs/nanogpt.
- Training commands require one allocated RTX 5090; no concurrent training on that GPU.
- Local CPU tests: C:/Users/Ahmed/anaconda3/python.exe (torch 2.8, CPU); plain `python` is not on PATH.

## This idea: vocabulary pruning (docs/vocab-prune.md; Notion 2026-11-05: drop unneeded tokens)
- The record already prunes: sampled softmax (record #360) trains steps 0-1106 over 10,240-24,576
  candidates (all microbatch targets + uniform stride negatives), not 50,304 ids; validation masks ids
  the tokenizer cannot emit after the previous token (canonical mask, #350). Validation forward is
  off the clock.
- Out of scope: retokenizing (changes the metric); choosing ids from validation counts (peeking).
  A removed id must keep finite probability: a validation token on it costs ~80 nats under -60 masking.
- CENSUS DONE 2026-11-10 (cluster logs/token-census/20261110T214058Z; tools/token_census.py replays
  the exact 328,663,040 training targets, CPU only): 49,989 of 50,257 ids occur; 268 never (mostly
  invalid-UTF-8 byte and control-char tokens), 1 validation token on them. Train counts < 10 / 100 /
  1000: 587 / 2,337 / 26,841 ids; validation hits 68 / 3,580 / 421,761. 99% coverage: 36,968 ids.
- EXACT MODE (mass 0, default): only never-seen ids leave, on steps 1107-1193. Never run; effect below noise.
- RANKED SWEEP (RTX_VOCAB_PRUNE_MASS=m: microbatch targets + most frequent ids so far, GPU-built, width
  min(record, ids covering 1-m), floor 6,144). RESULT 2026-11-11, seed-999 pairs at 74cf685 (loss / time):
  m=0.15 +0.5885 / -8.46%; 0.10 +0.5973 / -5.36%; 0.05 +0.5498 / -3.34%; 0.01 +0.3577 / -1.70%. Loss wrecked:
  ids outside the set are never negatives. ~10 s of the gain is GPU-side set building (port-only; user: none).
- STRATIFIED FIX (user, 2026-11-11): RTX_VOCAB_PRUNE_TAIL=t with a mass. RESULTS t=1024, seed-999 pairs
  at 2bf3d65 (loss / time): m=0.20 +0.0386 / -7.95%; 0.15 +0.0340 / -6.79%; 0.10 +0.0240 / -5.40%.
  All 12-19x the bar; ~0.005 loss per 1% time. EXACT END (RTX_VOCAB_PRUNE_EXACT_END=1, full softmax on
  1107-1193, 2411e4a): m=0.20 t=1024 +0.0238 / -6.24% (38% of the gap gone; rest is steps 0-1106).
  Exact end + t=2048 (width 8,192): +0.0093 / -4.60% (candidate 795.3 s ~ t=1024's; baseline drifted).
  Tail only m=0.20 t=4096: +0.0123 / +0.94% (one ~54 s spike in steps 926-950; user assumes node
  interference, so ~-5.5%). Exact end + t=4096: +0.0083 / -3.27%, only 0.001 better than t=2048: a
  ~+0.008 floor remains. STAGE GATE (RTX_VOCAB_PRUNE_FROM_STEP / _UNTIL_STEP: prune only from <= step < until,
  the record elsewhere), xe t=2048: 681-1106 +0.0035 / -3.89%; 965-1106 +0.0015 / -1.92%; 681-964 +0.0016 /
  -1.64% (each block alone inside the bar; they add up). Steps 0-680 stay the record (~0.006 for ~6 s).
  PER-STAGE MASS (RTX_VOCAB_PRUNE_SWITCH=step:mass:tail, 146e59f): from 681 + switch 965:0.10:2048 (width 13,184)
  = +0.0033 / -2.99%: no better than 8,192 everywhere, 7.6 s lost; exact-part width is not the lever.
  TAIL POWER (RTX_VOCAB_PRUNE_TAIL_POWER=b: tail weight (rest/t)^b, built 2026-11-14, not yet run): first pair
  b=0.5 from 681. Parked: frequency-weighted tail draw; user's pruned ending with an estimated normalizer.

## Commands
- Census: python -u tools/token_census.py (training venv; DATA_PATH if shards are elsewhere). Prints a
  short report and CENSUS_COMPLETE dir=logs/token-census/<UTC> (counts.npz, summary.json).
- Pair per point: python -u tools/run_pair.py --seeds 999 --label <label> --candidate-env RTX_VOCAB_PRUNE_MASS=<m>
  --candidate-env RTX_VOCAB_PRUNE_TAIL=<t> ... (knobs reach the gate + candidate only); full commands in docs/vocab-prune.md.
- GPU smoke: python -u tests/check_rtx5090.py --seed 999.
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
- Phase 1, seed 999: sampled-softmax taper +0.0064 / -1.7 s; per-head V/O whitening +0.012; whole Q/K
  whitening -0.001 (noise); layer smear -0.0009 / +4%; low-rank parallel MLP +0.0021 / -0.02% (paired).
- Eval-time sweep (temperature/windows): record settings already best. Async uploads -1.7% (local-only,
  kept on its own branch). Gram NS: little total-run potential.
- Phase 2 (paired, seed 999; experiment/attnres-last-init): zero-init block AttnRes +0.0369 / +10.5%;
  last-state init +0.0330 / +11.2%. Init explains at most ~0.004 of the regression.
- Phase 2 ideas (Notion 2026-11-05): AttnRes init (done), vocab pruning (this branch), mHC (not started).
  Notion "Ideas vs Results" rows: 1-2 AttnRes, 3-6 ranked, 7-9 stratified, 10-11 exact end, 12-13 tail 4096, 14-16 stage gate, 17 per-stage mass; notes = Setup / Result / Takeaway.
- Canonical-mask warnings and rebuild time remain part of measured training time.

## Logging
- Trainer log: logs/<UUID>.txt; launcher console: logs/run-*.console.txt.
- W&B entity/project i-learn/speedrun, uploaded only after training ends.
- Uploader: ~/venvs/wandb-upload/bin/python (WANDB_UPLOAD_PYTHON override).
- Paired names (Phase 2): baseline-for-<label>-1gpu-seed-N and <label>-1gpu-seed-N, set by the runner.
  Plain baseline-1gpu-seed-N = historical/unpaired baselines only. Names are display-only (IDs = log hash).
- Never expose API keys or upload raw source logs. Upload failure retains local results.

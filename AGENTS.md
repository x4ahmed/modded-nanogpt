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

## This idea: vocabulary pruning (docs/vocab-prune.md)
- Notion idea (2026-11-05): count how often each token appears in training data; drop unneeded ones.
- The record already prunes: sampled softmax (record #360) trains steps 0-1106 over 10,240-24,576
  candidates (all microbatch targets + uniform stride negatives), not 50,304 ids; validation masks ids
  the tokenizer cannot emit after the previous token (canonical mask, #350). Validation forward is
  off the clock.
- Full-softmax steps 1107-1193 (24.6M of 328.7M tokens): anchor record_late (P=24,576 on the taper,
  -1.7 s) -> ~0.1 s (FLOP bound ~0.4 s) per 1,000 ids removed; same-allocation time noise ~1-2 s.
- Out of scope: retokenizing (changes the metric); choosing ids from validation counts (peeking).
  A removed id must keep finite probability: a validation token on it costs ~80 nats under -60 masking.
- CENSUS DONE 2026-11-10 (cluster logs/token-census/20261110T214058Z; tools/token_census.py replays
  the exact 328,663,040 training targets, CPU only): 49,989 of 50,257 ids occur; 268 never (mostly
  invalid-UTF-8 byte and control-char tokens), 1 validation token on them. Train counts < 10 / 100 /
  1000: 587 / 2,337 / 26,841 ids; validation hits 68 / 3,580 / 421,761. 99% coverage: 36,968 ids.
- USER REQUEST 2026-11-10: implement pruning and run a pair to verify. Implemented (one GPU only):
  every microbatch marks its targets seen (GPU); steps 1107-1193 run the sampled CE kernels over a
  fixed 50,048-wide set = ids seen as targets so far + never-seen fill (exact for every target; C is
  rebuilt if a new id appears; full-softmax fallback if > 50,048 seen). Logs VOCAB_PRUNE lines.
  Prediction: time -0.1..+0.2 s, loss within noise. Pair not run yet; it is the 0% point of the sweep.
- RANKED SWEEP (RTX_VOCAB_PRUNE_MASS=m: microbatch targets + most frequent ids so far, GPU-built, width
  min(record, ids covering 1-m), floor 6,144). RESULT 2026-11-11, seed-999 pairs at 74cf685 (loss / time):
  m=0.15 +0.5885 / -8.46%; 0.10 +0.5973 / -5.36%; 0.05 +0.5498 / -3.34%; 0.01 +0.3577 / -1.70%.
  Loss is wrecked at every mass, even 0.01 (record widths on 93% of steps): ranked fill means ids
  outside the set are never negatives, so their logits stay high and steal probability at validation.
  m=0.01 still saved 14.7 s with a 1-5 s width ceiling: ~10 s of the gain is building candidates on
  the GPU instead of the host builder. Exact mode (m=0) never run. Next step is the user's call.

## Commands
- Census: python -u tools/token_census.py (training venv; DATA_PATH if shards are elsewhere). Prints a
  short report and CENSUS_COMPLETE dir=logs/token-census/<UTC> (counts.npz, summary.json).
- Pair per mass: python -u tools/run_pair.py --seeds 999 --label vocab-prune-m<m> --candidate-env
  RTX_VOCAB_PRUNE_MASS=<m> (run_pair gained --candidate-env here: gate + candidate only). Loop in
  docs/vocab-prune.md. Standalone runs self-name vocab-prune-m<m>-1gpu-seed-N.
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
- Phase 2 ideas (Notion, 2026-11-05): AttnRes init (done above), vocabulary pruning (this branch;
  census shows no measurable time lever), mHC (not started).
  Notion "Ideas vs Results" Phase 2 scoreboard: rows 1-2 = the AttnRes pairs. Its "Phase 2: Idea
  Notes" bullets carry What we did / Conclusion sub-bullets (AttnRes, pruning) and mHC "not started".
- Canonical-mask warnings and rebuild time remain part of measured training time.

## Logging
- Trainer log: logs/<UUID>.txt; launcher console: logs/run-*.console.txt.
- W&B entity/project i-learn/speedrun, uploaded only after training ends.
- Uploader: ~/venvs/wandb-upload/bin/python (WANDB_UPLOAD_PYTHON override).
- Paired names (Phase 2): baseline-for-<label>-1gpu-seed-N and <label>-1gpu-seed-N, set by the runner.
  Plain baseline-1gpu-seed-N = historical/unpaired baselines only. Names are display-only (IDs = log hash).
- Never expose API keys or upload raw source logs. Upload failure retains local results.

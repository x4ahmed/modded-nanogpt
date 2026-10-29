# Agent handoff

## Working rules
- Keep replies short, outcome first; normally under eight lines when practical.
- For coding work report changes, files, verification, and unresolved issues.
- Preserve the RTX adaptation and unrelated user changes. Do not restore old v1 plans.
- User permits commits and pushes to the current remote branch. Never force-push.
- Do not claim GPU validation from CPU tests. The user runs GPU checks on the cluster.
- Never SSH/scp to the cluster; give the user the command and ask them to paste the output.
- Update this file after meaningful changes, test results, or decisions, before handoff.
- Replace stale summaries rather than appending a diary. Keep under 100 lines.
- Keep only current settings, verified results, blockers, and next steps here.
- Put long logs in logs/ and test instructions in tests/README.md; never copy logs here.
- Do not silently change the model, precision, dataset, schedule, or validation metric.

## Goal and repository
- Establish a reproducible local single-RTX-5090 training baseline, then optimize it.
- Frozen local branch: baseline (5be1ba0); active: experiment/vo-heads, branched from baseline.
- Every idea gets its own branch from baseline and is tested alone; never merge or stack wins (user, 2026-10-30).
- Current adaptation is track_1_short (ANVIL2); original track_1_short is not disposable.
- Entrypoint: run_rtx.sh -> train_gpt_rtx.py; adaptation helpers: track_1_short/portable/.
- Cluster checkout: ~/speed-run/modded-nanogpt; venv: ~/venvs/nanogpt.
- Cluster uses Slurm; commands below require an existing one-GPU allocation.

## Configuration to preserve
- One RTX 5090 (SM120, 32 GB); observed torch 2.10.0+cu128 / CUDA 12.8.
- PORTABLE=1; RTX_MICROBATCH_TOKENS=16384; RTX_NGRAM_DIVISOR=32.
- N-gram table has 2,643,840 rows; this is a capacity reduction from upstream.
- Stage accumulation counts: 8/16/24/20/8; retain existing gradient scaling.
- RTX_EVAL_CE_SLAB_ROWS=4096; RTX_WARMUP_MICROBATCHES=2.
- Validated RTX_VAL_SEQ_LEN=65536. Default is 262144, NOT validated by this check.
- Keep validation packing fixed across local comparisons; do not claim upstream metric parity.
- Full default schedule: 1194 optimizer updates. Remove RTX_MAX_STEPS for full runs.
- Leave NUM_SCHEDULED_ITERATIONS unset; short tests truncate, never rescale the schedule.
- Seeds 999, 1000, 1001, 1002, 1003 completed; W&B verified 2026-10-21 (see docs/baseline-five-seeds.md).
- Training log is logs/<generated UUID>.txt; capture console separately for warnings.

## Last verified state (2026-10-21)
- Code commit f466280 fixed the observed fused-MLP NaNs on the user's 5090.
- Narrow 128-column MLP tiles use masked pointer stores, forward/backward, both layouts.
- FP8 math/scales and model structure were retained; wide-tile path retains descriptor stores.
- All nine diagnostic cases passed, including sampled/full loss forward and backward.
- MLP repeated eager/compiled forward and independent backward parity checks passed.
- tests/check_rtx5090.py passed all sampled warmup stages, two full updates, and validation.
- Validation loss after two updates: 8.883883 (not a convergence/baseline result).
- Peak allocated 20.01 GiB; reserved 24.57 GiB; reported headroom 2.00 GiB.
- Twenty-one CPU/Bash tests passed (runner, stores, uploader, launcher); no local test GPU.
- First full pilot completed: seed 999, launch commit 9f0dfb9, 1194/1194 updates.
- Pilot final validation loss 3.294856; recorded training time 837067 ms (13m57.067s).
- Pilot warmup 95s (separate); peak allocated 19.99 GiB, reserved 20.76 GiB, headroom 5.81 GiB PASS.
- Pilot log: logs/7911de55-2adc-4ef7-b53a-a2a5db3e2edb.txt; predates W&B integration.
- Supplied pilot output has finite logged losses and no canonical-mask fallback warning.
- Evidence of learning/completion, not proof of upstream parity or multi-seed stability.

## Open issue and next steps
- Active: per-head V/O whitening (optimizer only, transferable upstream). ANVIL whitens each head's V rows and O
  columns as their own matrix (optim/anvil.py vo_head_blocks); Q/K/MLP banks bitwise unchanged. docs/vo-heads.md.
- Value upstream: record prices val at ~164 ms/millinat; screen seed 999 vs baseline 3.294980/3.295181/3.295634, needs < -0.002.
- Log line VO_WHITENING; W&B vo-heads-1gpu-seed-<seed>. 2026-10-30: CPU suite passes; 5090 smoke passed (user report); seed 999 running.
- experiment/async-uploads (port-only host-sync fix), pair 2026-10-29: -14.15 s (-1.71%), loss unchanged. WIN;
  kept on its own branch (not merged). It does not transfer: the 8xH100 record already uploads this way.
- Sampled-softmax record_late (experiment/sampled-softmax-schedule), seed-999 pair: +0.0064 loss, -1.7 s. Stopped.
- Gated DeltaNet (experiment/gated-deltanet, 83fda24) five seeds: +0.0041098 loss (CI +0.0023..+0.0059), +8.17% time. Not promoted.
- Screening rule (2026-10-22): one seed per idea; five seeds only if it beats 3.2949. W&B deltanet-l5-1gpu-seed-<seed>.
- Finished, not promoted (5 seeds each, vs 3.2949 / 861.6 s): block AttnRes +0.036 / +10%; minimal AttnRes readout
  +0.010 / +0.6%; bigram output prior +0.0025 / +0.85%. Notion Ideas page has the scoreboard.
- Smoke test's canonical-mask worker failed with wait status 9; full pilot did not reproduce it.
- canonical_mask.py wait() can SIGKILL after 60 seconds; cause not proven to be host OOM.
- Mask wait/rebuild is timed. Smoke run's 281930 ms is not clean optimizer throughput.
- Continue tracking mask warnings; never silently exclude mask overhead from timings.
- Five-seed baseline at 5be1ba0: loss 3.2948572 ± 0.00104874; training 861.635 ± 1.608645 s (sample SD).
- All five completed 1194 updates, memory gate passed, no reported mask fallback; pilots excluded.
- Compare candidates on the same seeds/settings; session variance is not cross-session variance.
- Notion: Experiment 1-4 pages under Progress; Ideas scoreboard rows 1-4 (5-seed means), row 5 one seed (2026-10-28).
- Mean loss is 0.0148572 above upstream ≤3.28; no seed meets target. Capacity/eval differences are not isolated.

## Verification commands
- GPU diagnostics: python -u tests/diagnose_rtx5090.py
- GPU smoke: python -u tests/check_rtx5090.py
- CPU: python -m unittest discover -s tests -p 'test_*.py' -v
- Diagnostic cases are isolated; BF16 controls are tests, not production configuration.
- Keep diagnostic timings separate from uninstrumented full training timings.

## Post-training monitoring
- tools/upload_wandb.py imports completed logs only; training code/timer unchanged.
- Setup/API-key guidance: tools/README.md; use a separate uploader venv, no GPU needed.
- Preview with --dry-run; never upload raw source logs or put API keys in Git/chat.
- Compare recorded training_time_ms, not W&B uploader runtime. Do not infer training commit/seed.
- Upload one log per run; stable content-based IDs refuse duplicate online imports.
- User screenshot confirms W&B upload succeeded. Uploader venv required pip installation via training pip --python.
- Future uploads plot training loss vs optimizer step, training seconds and minutes; axis-only plots hidden.
- Loss/time uses each training-loss line's timestamp, not later validation/finalization at the same step.
- Final loss, total time and memory remain summary metrics; old workspace panels need manual hiding.
- run_rtx.sh auto-uploads to i-learn/speedrun only after successful non-preflight runs.
- Uses ~/venvs/wandb-upload/bin/python (override WANDB_UPLOAD_PYTHON); RTX_WANDB_UPLOAD=0 opts out.
- Launcher saves console log and captures launch commit/seed; upload failure retains local results.
- Launcher names W&B runs baseline-1gpu-seed-<seed> (vo-heads-... on this branch); repeats keep distinct run IDs.

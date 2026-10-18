# Agent handoff

## Working rules
- Keep replies short, outcome first; normally under eight lines when practical.
- For coding work report changes, files, verification, and unresolved issues.
- Preserve the RTX adaptation and unrelated user changes. Do not restore old v1 plans.
- User permits commits and pushes to the current remote branch. Never force-push.
- Do not claim GPU validation from CPU tests. The user runs GPU checks on the cluster.
- Update this file after meaningful changes, test results, or decisions, before handoff.
- Replace stale summaries rather than appending a diary. Keep under 100 lines.
- Keep only current settings, verified results, blockers, and next steps here.
- Put long logs in logs/ and test instructions in tests/README.md; never copy logs here.
- Do not silently change the model, precision, dataset, schedule, or validation metric.

## Goal and repository
- Establish a reproducible local single-RTX-5090 training baseline, then optimize it.
- Branch: rtx-local-v2. Remote: https://github.com/x4ahmed/modded-nanogpt.
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
- Notion's existing seed plan is 999, 1000, 1001, 1002, 1003; 999 complete, four pending.
- Training log is logs/<generated UUID>.txt; capture console separately for warnings.

## Last verified state (2026-10-18)
- Code commit f466280 fixed the observed fused-MLP NaNs on the user's 5090.
- Narrow 128-column MLP tiles use masked pointer stores, forward/backward, both layouts.
- FP8 math/scales and model structure were retained; wide-tile path retains descriptor stores.
- All nine diagnostic cases passed, including sampled/full loss forward and backward.
- MLP repeated eager/compiled forward and independent backward parity checks passed.
- tests/check_rtx5090.py passed all sampled warmup stages, two full updates, and validation.
- Validation loss after two updates: 8.883883 (not a convergence/baseline result).
- Peak allocated 20.01 GiB; reserved 24.57 GiB; reported headroom 2.00 GiB.
- Seventeen CPU/Bash tests passed (runner, stores, uploader, launcher); no local test GPU.
- First full pilot completed: seed 999, launch commit 9f0dfb9, 1194/1194 updates.
- Pilot final validation loss 3.294856; recorded training time 837067 ms (13m57.067s).
- Pilot warmup 95s (separate); peak allocated 19.99 GiB, reserved 20.76 GiB, headroom 5.81 GiB PASS.
- Pilot log: logs/7911de55-2adc-4ef7-b53a-a2a5db3e2edb.txt; predates W&B integration.
- Supplied pilot output has finite logged losses and no canonical-mask fallback warning.
- Evidence of learning/completion, not proof of upstream parity or multi-seed stability.

## Open issue and next steps
- Smoke test's canonical-mask worker failed with wait status 9; full pilot did not reproduce it.
- canonical_mask.py wait() can SIGKILL after 60 seconds; cause not proven to be host OOM.
- Mask wait/rebuild is timed. Smoke run's 281930 ms is not clean optimizer throughput.
- Continue tracking mask warnings; never silently exclude mask overhead from timings.
- Keep training code/settings fixed; finish seeds 1000-1003. Report mean and sample SD (n-1).
- One full pilot exists; no statistically established v2 baseline yet.
- Notion root, Progress, Overview, Planning, Baseline and Sarmata now reflect the successful pilot.
- Progress seed-results table: https://www.notion.so/3cb0b758a24e81faa372fac85828c1a0 (1/5 complete).

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
- Parser checked against the user's smoke output; live W&B upload awaits user authentication.
- run_rtx.sh auto-uploads to i-learn/speedrun only after successful non-preflight runs.
- Uses ~/venvs/wandb-upload/bin/python (override WANDB_UPLOAD_PYTHON); RTX_WANDB_UPLOAD=0 opts out.
- Launcher saves console log and captures launch commit/seed; upload failure retains local results.

# Agent handoff

## Working rules
- Keep replies short, outcome first; normally under eight lines when practical.
- For coding work report changes, files, verification, and unresolved issues.
- Preserve the RTX adaptation and unrelated user changes. Do not restore old v1 plans.
- User permits commits and pushes to the current remote branch. Never force-push.
- Do not claim GPU validation from CPU tests. The user runs GPU checks on the cluster.
- Never SSH/scp to the cluster; give the user commands and ask for the output.
- Update this file after meaningful changes, test results, or decisions, before handoff.
- Replace stale summaries rather than appending a diary. Keep under 100 lines.
- Keep only current settings, verified results, blockers, and next steps here.
- Put long logs in logs/ and test instructions in tests/README.md; never copy logs here.
- Do not silently change the model, precision, dataset, schedule, or validation metric.

## Goal and repository
- Establish a reproducible local single-RTX-5090 training baseline, then optimize it.
- Branch: experiment/gram-ns-mlp, from frozen baseline (5be1ba0). Remote: x4ahmed/modded-nanogpt.
- Every idea branches directly from baseline; do not stack earlier experiments (user, 2026-10-30).
- Current goal: speed gains on both one RTX 5090 and 8xH100, with comparable validation loss.
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
- Baseline seeds 999, 1000, 1001, 1002, 1003 are all complete; pilots are excluded.
- Training log is logs/<generated UUID>.txt; capture console separately for warnings.

## Current experiment and verification (2026-10-30)
- MLP-only paired Gram Newton-Schulz; docs/gram-ns-mlp.md. Six ANVIL maps in three pairs.
- Retains BF16, coefficients, initial Gram/trace normalization, momentum, equalizer, decay and grouping.
- Rebuilds the Gram every two maps. Fused symmetric products include diagonal shifts before BF16 rounding.
- ANVIL_MLP_WHITENING=gram_pair is the branch default; standard restores the original computation.
- The same per-bank switch serves both trainers. Q/K, V/O and Adam stay on the baseline paths.
- Model, loss, data, schedule, validation and communications are unchanged. Numerical updates are not bitwise equal.
- 36 CPU/Bash tests pass, including SVD/polynomial references, momentum, routing, AOT mutation handling,
  uploader metadata, launcher names and failure/OOM status handling.
- New fused kernel compiled offline for SM86/90/120 using temporary Triton Windows 3.4.0.post21.
- Six fused-kernel cases passed on local RTX 3050 / torch 2.8: odd sizes, transpose strides, beta=0/NaN C.
- Local full optimizer diagnostic could not run: Windows C compiler/CUDA toolkit missing. Direct CUDA-driver
  kernel tests bypassed that launcher limitation. RTX 5090/8xH100 runtime and convergence are NOT validated.
- tests/check_gram_ns.py exercises full-size numerical cases, graph replay/reset/scalars, frozen slots and
  interleaved whole-bank timing. Batch 3 simulates local H100 shard shape, without collectives; batch 24 is RTX.

## Open issue and next steps
- Smoke test's canonical-mask worker failed with wait status 9; full pilot did not reproduce it.
- canonical_mask.py wait() can SIGKILL after 60 seconds; cause not proven to be host OOM.
- Mask wait/rebuild is timed. Smoke run's 281930 ms is not clean optimizer throughput.
- Continue tracking mask warnings; never silently exclude mask overhead from timings.
- Next: cluster Gram-NS diagnostic, then RTX smoke; paired seed-999 standard/gram_pair full runs if both pass.
- Five seeds only after a promising screen. Bank speed is not total training speed; H100 needs separate testing.
- Frozen baseline: loss 3.2948572 +/- 0.00104874; time 861.635 +/- 1.608645 s (sample SD, five seeds).
- Compare within the same session/settings; older elapsed times are not a same-session control.
- Prior ideas not promoted: block/minimal AttnRes, bigram prior, gated DeltaNet, record_late softmax taper,
  per-head V/O whitening (+.0124 loss, no speed gain). Async uploads won locally but is RTX-port-only.
- Notion scoreboard: https://www.notion.so/3b70b758a24e80a48b1ad83a7b776d1b

## Verification commands
- Optimizer diagnostic: python -u tests/check_gram_ns.py (prints progress; logs/check-gram-ns-*.txt).
- GPU smoke: python -u tests/check_rtx5090.py --mlp-whitening gram_pair
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
- W&B names: gram-ns-mlp-1gpu-seed-<seed>; standard mode uses baseline-1gpu-seed-<seed>.
- Whitening mode/precision/maps/refresh interval are logged and uploaded as configuration.

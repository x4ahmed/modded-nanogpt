# Agent handoff

## Working rules
- Keep replies short and outcome first. Never omit failed checks or material limits.
- Preserve the RTX adaptation and unrelated user changes. Do not restore old v1 plans.
- User permits commits and pushes. Never force-push or move the frozen baseline.
- Every experiment branches directly from baseline; never stack or merge experiments.
- NEW USER RULE (2026-11-01): run a fresh baseline alongside every candidate, using
  the same seed, GPU allocation, environment, data and validation settings. Future
  commands must produce paired runs; historical results alone are not the control.
- Start with seed 999; expand promising ideas to fresh paired seeds 999-1003.
- Never SSH/scp to the cluster. The user executes GPU commands and shares output.
- CPU checks are not GPU validation. No speed/quality claim without full-run evidence.
- Update this file after meaningful changes/results/decisions. Replace stale text;
  keep under 100 lines. Put protocols in docs/ and tests/, large results in logs/.

## Repository and goal
- Goal: reduce training time at matched validation quality on RTX 5090, with a
  plausible transfer path to 8xH100. Actual multi-GPU validation remains necessary.
- Frozen baseline: 5be1ba0ca09293a007997cdce51c036bca3e89ae.
- Active: experiment/low-rank-parallel-mlp, directly from baseline.
- Entry: run_rtx.sh -> train_gpt_rtx.py; shared architecture: track_1_short/.
- Cluster checkout: ~/speed-run/modded-nanogpt; training venv: ~/venvs/nanogpt.
- Commands require one allocated RTX 5090; no concurrent training on that GPU.

## Low-rank experiment
- Only the extra MLP at position 8, bank slot 11, changes (zero-based indices).
- Down projection 2816->768 becomes 2816->384->768, no additional nonlinearity.
- Up projection, other ten active MLPs, attention, residual paths and schedules stay.
- Both down factors train with ANVIL every step (2x down LR multiplier plus existing
  shape scaling). Each uses a full-matrix averaged all-reduce on multiple GPUs.
- A starts zero, B random nonzero; initialization preserves the baseline RNG stream.
- Old slot-11 down weight remains frozen zero padding. No allocated-memory saving claim.
- Training keeps FP8 GEMMs; validation BF16. New bottleneck uses current amax scaling.
- Both factors participate in warmup reset, tail averaging, factorwise decontraction
  and FP8-refresh capture. Optimization/averaging of factors differs from dense weights.
- docs/low-rank-parallel-mlp.md owns details and limitations.
- Theoretical saving: 18.2% of this MLP's matrix arithmetic, ~1.65% across all MLPs;
  first paired run shows no useful end-to-end gain (results below).

## Frozen local settings
- One RTX 5090 (SM120, 32 GB); PORTABLE=1.
- RTX_MICROBATCH_TOKENS=16384; RTX_NGRAM_DIVISOR=32 (2,643,840 n-gram rows).
- Stage accumulation counts 8/16/24/20/8; retain existing gradient scaling.
- RTX_VAL_SEQ_LEN=65536; RTX_EVAL_CE_SLAB_ROWS=4096.
- RTX_WARMUP_MICROBATCHES=2; RTX_MIN_HEADROOM_GIB=1.5.
- Full schedule: 1194 optimizer updates. Unset RTX_MAX_STEPS, RTX_PREFLIGHT and
  NUM_SCHEDULED_ITERATIONS. Short tests truncate, never rescale schedules.
- Keep local validation packing fixed; do not claim upstream metric parity.

## Commands and results
- Pair: python -u tools/run_low_rank_pair.py --seeds 999 (clean experiment checkout).
- Runner gates on MLP check + optimizer check + full-model smoke, runs baseline then candidate,
  saves commits/results/deltas in logs/pairs/<UTC>/seed-999.json, restores experiment.
- GPU checks: tests/check_low_rank_mlp.py, tests/check_low_rank_optimizer.py,
  then tests/check_rtx5090.py (each via python -u).
- CPU: python -m unittest discover -s tests -p 'test_*.py' -v.
- Current CPU checks cover algebra, AOT autograd, initialization/RNG, reset, replica
  ownership/reduction, pair ordering/failure handling and existing launch/upload behavior.
- Focused check fixes: detach FP8 caches like production; use production scale constants.
- Focused MLP PASS: oracle output error 0.0309%, largest tensor-gradient error
  0.0683%, scalar exact, BF16 eval exact (eager and compiled).
- Fix: rank-384 XTX on limited-shared-memory GPUs uses two pipeline stages (was four).
  Tiles, precision and optimizer equations unchanged; baseline/H100 launches preserved.
  Offline Triton 3.6 SM120 compile reproduces 114688 -> 49152 bytes; cluster pair
  subsequently completed at 52a1934. Full CPU suite: 43 tests passed.
- Gates unchanged: quantized oracle 3%, BF16 output/gradients 10%/15%, aligned scalar
  15%. Random scalar BF16 drift is diagnostic (cancellation); quantized scalar gates.
  All parity failures print before a nonzero exit; full runs remain gated.
- User-supplied seed-999 pair 20261101T004642Z complete: baseline 5be1ba0 loss
  3.295196 / 835.291 s; candidate 52a1934 loss 3.297256 / 835.086 s.
  Delta: loss +0.002060, time -0.205 s (-0.02454%); memory PASS for both.
  No useful speedup demonstrated; recommend retaining baseline, no five-seed expansion
  of this configuration yet. One pair cannot establish variance or significance.
  Full-model peak allocated: baseline 19.99 / candidate 20.02 GiB. No H100 test.
- Frozen five-seed historical baseline: loss 3.2948572 +/- 0.00104874 SD;
  time 861.635 +/- 1.608645 s SD. Use fresh paired controls for new comparisons.
- Prior architecture/optimizer ideas were not promoted: AttnRes, minimal readout,
  bigram output prior, DeltaNet, sampled taper, per-head V/O and whole Q/K whitening.
- Async uploads showed a local-only timing gain; Gram NS showed little total-run potential.
- Canonical-mask warnings and rebuild time remain part of measured training time.

## Logging
- Trainer log: logs/<UUID>.txt; launcher console: logs/run-*.console.txt.
- W&B entity/project i-learn/speedrun, uploaded only after training ends.
- Uploader: ~/venvs/wandb-upload/bin/python (WANDB_UPLOAD_PYTHON override).
- Names: baseline-1gpu-seed-N and low-rank-parallel-r384-1gpu-seed-N.
- Repeated names retain distinct IDs; pair JSON records corresponding console logs.
- Never expose API keys or upload raw source logs. Upload failure retains local results.

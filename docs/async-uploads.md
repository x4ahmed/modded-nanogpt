# Async uploads (port-only systems fix)

Branch `experiment/async-uploads`, from `baseline` (5be1ba0). The model, data, schedule and loss math
are unchanged: the GPU runs the same kernels on the same values in the same order. Only when the CPU
waits for the GPU changes.

## What was wrong

1. **A GPU drain before every sampled microbatch.** `portable/sampled.py` uploaded each candidate set
   with three plain `copy_()` calls from pageable host memory. A blocking host-to-device copy in PyTorch
   is `cudaMemcpyAsync` plus `cudaStreamSynchronize`: the CPU waits until the GPU has finished all queued
   work, then the GPU idles while Python launches the next forward. This happened before 18,560 of the
   run's 20,060 microbatches (steps 0-1106 use sampled softmax). The record avoids it with pinned slots
   and non-blocking copies (`perf/sampled_softmax_overlap.py`); the port had dropped them.
2. **A GPU drain at the start of every 24- and 20-microbatch step.** The loader uploads all of a step's
   microbatches before its first forward, through the record's ring of 16 pinned slots (sized for one
   batch per rank per step). In stage 2 and the taper (493 steps), the 17th upload waited for the GPU to
   finish the previous step, and the GPU then idled while the CPU loaded the rest.

Local checks (RTX 3050 laptop, torch 2.8): a pageable `copy_()` blocked the CPU for all 74.4 s of queued
GPU work; a pinned non-blocking copy returned after 0.07 ms. The new `prepare()` (Triton transpose
replaced by a copy) ran 12 microbatches with the CPU ahead of a busy GPU, reusing each slot twice:
every microbatch's device buffers matched its own build exactly.

## The fix

- `portable/sampled.py`: the three copies go through a ring of 4 pinned slots (the record's `PinnedSlot`
  and `PINNED_SLOTS`) with `non_blocking=True`, on the compute stream. Each slot's event is recorded
  after its copies; the CPU rewrites a slot only after that event.
- `perf/pinned_batches.py` takes a ring depth (default 16, the record's). The port passes
  `BATCH_STAGING_SLOTS` from `portable/config.py`: two steps of microbatches (48 at 16,384 tokens).
- The run log prints `ASYNC_UPLOADS batch_slots=48 candidate_slots=4`. Baseline logs have no such line.
- W&B runs from this branch are named `async-uploads-1gpu-seed-<seed>`.

## Expectation and screening

About 8-12 s faster (~1%), with the loss unchanged up to run-to-run noise (seed 999 gave 3.294980 and
3.295181 on the baseline). Port-only: the upstream record already uploads this way.

Screen with one pair in the same allocation, seed 999: the `baseline` branch, then this branch. It is a
win if the time drops by more than ~5 s (3x the 1.6 s within-session SD) and the loss stays within
±0.002 of the paired baseline run.

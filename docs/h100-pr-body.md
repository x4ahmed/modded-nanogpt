<!-- PR title: Track 1: Stratified sampled softmax on the widest steps (-0.53s, -1.32% same hardware, n=18) -->

## Summary

On steps 965-1106, where record #92's sampled softmax is widest (24,576 ids), train on 8,192 columns instead. Everything else is the record's.

![Stratified sampled softmax: the schedule, how a pruned step builds its sets, and the measured step times](https://raw.githubusercontent.com/x4ahmed/modded-nanogpt/c9384a80be5d900037bcbd6bcbdca347fadce942/records/track_1_short/2026-11-15_StratifiedSoftmaxTail/stratified_softmax.png)

## Result

| Run | N | Time (s) | Validation loss | p (mean ≤ 3.28) |
|---|---:|---:|---:|---:|
| #92, same node, interleaved | 9 | 40.235 ± 0.017 | 3.27608 ± 0.00136 | |
| This PR | 18 | 39.702 ± 0.042 | 3.27670 ± 0.00114 | 3.5e-10 |

**0.533 s faster (1.32%)** at +0.62 millinats (standard error about 0.5). Every run of this PR is faster than every baseline run.

## How it works

Each GPU's 49,152-row loss is split into 3 chunks of 16,384 rows (the model still runs once). Each chunk scores 6,144 exact ids (its targets plus this GPU's most frequent targets so far) and 2,048 random ids weighted ×21.6 to stand for all the ids left out, which keeps the normalizer unbiased. The CE kernel is the record's with one log weight per column.

## Loss buffer

Even with the loss cost two standard errors higher, this is about 320 ms per millinat, above the 120-250 ms that #360 measures.

## Disclosures

- Compile caches were not wiped between legs. This tree's 3 cold legs were slower (39.78 s) than its 15 warm ones (39.69 s); on cold legs only, the win is 0.45 s.
- The first run on the node was declared a functional check beforehand and is not counted (3.2757, 39.748 s).
- The frequency counts are gathered inside the timed run; nothing is precomputed from the data.
- Based on master alone. Not tested together with #371, which samples rows where this samples columns.

## Size and files

One new 151-line module and about 230 changed lines in 8 files. Write-up, 27 logs and scripts: `records/track_1_short/2026-11-15_StratifiedSoftmaxTail/`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

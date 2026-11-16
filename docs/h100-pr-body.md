<!-- PR title: New record attempt: stratified sampled softmax on the widest steps (39.70 s, 0.53 s faster than #92 on the same node) -->

One change on top of record #92 (ANVIL2, #360): steps 965-1106, where the record's sampled softmax is widest (24,576 ids), train on 8,192 columns instead.

![Stratified sampled softmax: the schedule, how a pruned step builds its sets, and the measured step times](https://raw.githubusercontent.com/x4ahmed/modded-nanogpt/c9384a80be5d900037bcbd6bcbdca347fadce942/records/track_1_short/2026-11-15_StratifiedSoftmaxTail/stratified_softmax.png)

## Result

18 runs of this tree interleaved with 9 runs of record #92 (`4ea6b93`) on the same 8xH100 node in the same session; every run counted.

| | runs | wall (training section) | final val CE |
|---|---|---|---|
| **this PR** | **18** | **39.702 ± 0.042 s** | **3.27670 ± 0.00114** |
| record #92 (same node, interleaved) | 9 | 40.235 ± 0.017 s | 3.27608 ± 0.00136 |
| **win** | | **0.533 s (1.32%)** | +0.62 millinats |

- Rule 2: one-sided t against the 3.28 gate, t = −12.3, p = 3.5e-10.
- Rule 4: every run of this PR (39.659–39.809 s) is faster than every baseline run (40.200–40.257 s); one-sided Welch p = 1.3e-25.
- The saving sits in the changed steps: a pruned step takes 44.6 ms against 48.4 ms; the other 1044 steps differ by 0.04 s in total.
- Record #92 is 39.914 s as published; it measured 40.235 s on this node. This PR is below both.

## The change

On steps 965-1106 each GPU's 49,152-row microbatch is split into 3 chunks of 16,384 rows for the loss only (the model still runs once). Each chunk gets its own candidate set:

- **6,144 exact ids** at weight 1: every target in the chunk, filled up with the ids this GPU has seen most often as targets so far in the run.
- **2,048 random ids** drawn uniformly from all the remaining ids, each at weight (remaining ids) / 2,048 ≈ 21.6.

The loss normalizes over `sum_exact exp(z) + w * sum_tail exp(z)`, an unbiased estimate of the full-vocabulary normalizer, so rare ids are still pushed down with the right strength on average. The CE kernel is the record's with one log weight per column added inside its three `exp()` calls. Chunks are needed because every target must be in its set: a 49,152-row microbatch holds up to 10,397 distinct targets, a 16,384-row chunk at most about 5,300.

Everything else is the record's: the other steps, the schedule, the optimizer, the data and the validation (full softmax, unchanged).

## Loss buffer

0.533 s for +0.62 millinats (standard error about 0.5). Even with the loss cost two standard errors higher, that is about 320 ms per millinat, above the 120–250 ms per millinat that record #92's write-up measures. On a one-GPU port of the record, paired runs showed the same thing directly: this change cost +1.0 millinats for −2.1% time, while cutting 26 steps cost +4.3 millinats for −1.9%.

## Size

One new 151-line module (`track_1_short/stratified_softmax.py`) and about 230 changed lines in 8 existing files, plus 13 CPU tests. `STRATIFIED_SOFTMAX=0` turns the change off.

## Disclosures

- Compile caches were not wiped between legs (record #92's pool wiped them); the timed section starts after compilation, warmup and graph capture either way.
- The first execution of this tree on the node was declared a functional check beforehand and is not in the pool (3.2757, 39.748 s).
- The tail draws use their own numpy generator seeded per GPU with a fixed constant; no global RNG stream is touched and the runs are otherwise unseeded.
- The frequency ranking is counted during the run, from step 0; nothing is precomputed from the data.
- Warmup and graph capture have the same shape as record #92 (25 untimed warmup steps, 12 captured step graphs); peak memory is unchanged.

The full write-up, the 27 logs, the pool script and the statistics script are in `records/track_1_short/2026-11-15_StratifiedSoftmaxTail/`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

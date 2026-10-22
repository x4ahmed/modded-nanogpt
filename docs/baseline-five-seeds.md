# RTX local baseline: five seeds

Fetched read-only from W&B project `i-learn/speedrun` on 2026-10-21.
Runs uploaded 2026-10-20; pilots excluded. Training commit for all five:
`5be1ba0ca09293a007997cdce51c036bca3e89ae`.

| Seed | Final validation loss | Recorded training seconds | W&B run ID |
| --- | --- | --- | --- |
| 999 | 3.294980 | 859.401 | import-3502dc8234b9b98510ba28b7 |
| 1000 | 3.295021 | 862.255 | import-eb9b0651347ed8bed3cada94 |
| 1001 | 3.295319 | 862.155 | import-2d531c67ccb1f5920ee37f0a |
| 1002 | 3.295873 | 860.749 | import-5c82098f12cdd66d1abe37b9 |
| 1003 | 3.293093 | 863.615 | import-6f6df5b215c8f17cf6a732bf |

## Statistics

Sample variance and standard deviation use denominator n-1 (n=5).

| Metric | Mean | Sample SD | Sample variance | SD / mean |
| --- | --- | --- | --- | --- |
| Validation loss | 3.2948572 | 0.00104874 | 0.0000010998552 | 0.03183% |
| Training seconds | 861.635 | 1.608645 | 2.587738 s² | 0.18670% |

All runs finished 1194 updates, passed the memory gate, and reported no canonical-mask
fallback. Peak allocated memory was 20465 MiB in each run. Uploaded configurations
match: RTX 5090, microbatch 16384 tokens, n-gram divisor 32 (2643840 rows), validation
sequence length 65536, CE slab 4096 rows, full schedule.

This is a stable initial local baseline, not evidence of parity with the upstream record.
Five sequential runs estimate variation in this session, not across all cluster conditions.
Compare candidates using the same five seeds and settings; small changes need paired
replication. Recorded training time excludes compilation/warmup and post-training upload;
it is not total job wall time. W&B summaries were verified; raw cluster logs were not
retrieved because SSH was unavailable.

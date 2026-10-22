# Block AttnRes: five-seed results

Fetched from W&B i-learn/speedrun on 2026-10-22. Candidate commit: 9655fc3aac633dfc8063a7130430921e4c2661ba.
Baseline: 5be1ba0ca09293a007997cdce51c036bca3e89ae. Block size 6, injections on.

| Seed | Baseline loss | Candidate loss | Baseline seconds | Candidate seconds | W&B run |
| --- | --- | --- | --- | --- | --- |
| 999 | 3.294980 | 3.332434 | 859.401 | 943.668 | https://wandb.ai/i-learn/speedrun/runs/import-cbe4332a4525538f3eb0b132 |
| 1000 | 3.295021 | 3.327721 | 862.255 | 948.338 | https://wandb.ai/i-learn/speedrun/runs/import-6b6fde979f0251bb7314fb5c |
| 1001 | 3.295319 | 3.334573 | 862.155 | 950.957 | https://wandb.ai/i-learn/speedrun/runs/import-072097fd160792b1799d7b12 |
| 1002 | 3.295873 | 3.329621 | 860.749 | 948.445 | https://wandb.ai/i-learn/speedrun/runs/import-1fa19ba7006f5edf380fceef |
| 1003 | 3.293093 | 3.329656 | 863.615 | 950.947 | https://wandb.ai/i-learn/speedrun/runs/import-0a66a1d2b421aed289c046db |

Candidate mean/sample SD: loss 3.330801 / 0.0026965023; time 948.471 / 2.97480865 seconds.
Sample variance (n-1): loss 0.0000072711245; time 8.8494865 s².
Paired mean differences: +0.0359438 loss; +86.836 seconds (+10.078% time).
All five are finished, completed 1194 updates, memory gate passed, no reported canonical-mask fallback.
Peak allocation 20465 MiB in each run, same as baseline. Configs agree on RTX 5090,
microbatch 16384, ngram divisor 32, val sequence 65536, eval CE slab 4096 and full schedule.

Conclusion: do not promote this candidate. Every seed has worse quality and longer recorded time.
Partial isolation prevents blaming AttnRes alone. Consider the narrower static arm or isolated integration
changes; profile before assigning slowdown to depth softmax. Timings are across sessions, not interleaved.
W&B summaries/configs verified; raw cluster logs not newly retrieved. No evidence of better validation at step 900.
Notion Progress and experiment page updated with the table and reflection.

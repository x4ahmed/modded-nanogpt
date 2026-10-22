# Minimal AttnRes: depth-attention readout (awaiting GPU validation)

Branch `experiment/minimal-attnres`, created from `baseline` (`5be1ba0`). Not built on
`experiment/attention-residuals`, whose block-mode candidate regressed (+0.036 loss, +10% time).

## What changes

`RTX_READOUT=attnres` replaces only the post-loop MUDD mix. Every layer, residual scaling, injection,
skip, cache, value embedding and the last layer's MUDD coefficients are the unchanged baseline.

| | Baseline (`RTX_READOUT=mudd`, default) | Candidate (`RTX_READOUT=attnres`) |
| --- | --- | --- |
| Readout | `x + sum_k (mu_k + delta_k) * src_k`, per-token MUDD coefficients and channel-group deltas | `sum_k softmax_k(<v_k / rms(v_k), q> + b_k) * v_k` |
| Sources | x plus 10: cache[0], cache[7], cache[9], ve[1], cache[3], ve[2], ve[10], ve[8], last MLP input, norm(cache[7]) | the same 11 (x first) |
| New parameters | none | `readout_query` (768, zero init), `readout_bias` (11) |

Adapted from the [official AttnRes pseudocode](https://github.com/MoonshotAI/Attention-Residuals):
RMS-normalized keys, unnormalized values, softmax over depth per token, no mixing across positions.

**Deviation: the bias.** The paper starts from uniform weights (zero query). Here a learned per-source
logit bias starts x at weight 0.9 and the other ten at 0.01 each, close to the baseline mix's start
(dominated by x). Reason: the block-mode run started every layer from a uniform average, a likely part
of its regression. Both readout parameters use replicated Adam, base LR 0.008 × 1.0, betas (0.9, 0.99),
no weight decay, fp32. The post-loop MUDD parameters (`mudd_*[1]`, `mudd_w2g`) stay allocated and get
no gradient in this mode.

Cost: one depth mix per forward instead of the MUDD MLP; expected near time-neutral.
A result compares AttnRes with MUDD as the final readout only, not AttnRes as layer-input routing.

## Commands

Cluster smoke check first (one allocated RTX 5090): see the command block in `tests/README.md`,
`python -u tests/check_rtx5090.py --readout attnres --seed 999`. Then five seeds with
`RTX_READOUT=attnres`; W&B names are `readout-attnres-1gpu-seed-<seed>`. Compare paired with the
baseline's five seeds (loss 3.2948572 ± 0.00104874, time 861.635 ± 1.608645 s at `5be1ba0`).

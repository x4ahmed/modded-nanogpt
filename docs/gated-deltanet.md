# Gated DeltaNet on one layer (awaiting GPU validation)

Branch `experiment/gated-deltanet`, created from `baseline` (`5be1ba0`). `RTX_DELTANET=1` enables it;
the default `0` is the unchanged baseline. Code: `track_1_short/model/gated_deltanet.py`.

## What changes

Layer 5's sliding-window softmax attention (default; `RTX_DELTANET_LAYER=0|2|5`, the narrow
short-window layers) becomes Gated DeltaNet ([arXiv:2412.06464](https://arxiv.org/abs/2412.06464)):
a linear-attention memory per head that is decayed, then corrected toward each new key/value
(the delta rule), and reset at every document boundary. It sees the whole document instead of a
short window. Everything else is the baseline.

| Part | Setting |
| --- | --- |
| q, k, v, o projections | The layer's own `qk_bank` / `vo_bank` slices (ANVIL), bf16 instead of fp8 for this layer |
| Head sizes | 6 heads, key 64, value 128 (the layer's narrow widths) |
| Short conv | Depthwise causal, 4 taps, SiLU; starts as identity; does not cross documents |
| Decay / write strength | Per head: `g = -exp(A_log) * softplus(a + dt_bias)`, `beta = sigmoid(b)` |
| Output | Per-head RMSNorm, times `silu(gate)`, then the layer's O projection and output gains |
| New parameters | `gdn_conv`, `gdn_ab`, `gdn_A_log`, `gdn_dt_bias`, `gdn_gate`, `gdn_o_norm`: Adam, LR 0.008, betas 0.9/0.99, no weight decay, fp32 |
| Init | Deterministic decay spread over heads; gate/ab from a private generator (global RNG untouched, so all other inits match the baseline) |
| Dropped on this layer | Rotary, QK-norm (L2 norm instead), paired heads, sliding window |

Kernel: `fla.ops.gated_delta_rule.chunk_gated_delta_rule` (flash-linear-attention, Triton), wrapped as a
torch custom op so the compiled model stays one graph. The backward recomputes this layer's forward
(one extra layer forward per step). Zero-length padding segments are dropped before the call.

## Expectations

Judge on loss at 1194 updates, one seed first. Expect slower steps: bf16 instead of fp8 for this layer,
the recompute, and an untuned kernel on SM120. A loss win would justify fusing later; a time loss
alone would not reject it.

## Setup and commands

The cluster venv needs flash-linear-attention once (the run logs its version):

    pip install flash-linear-attention

GPU smoke, which also checks the fla kernel against a plain-torch recurrence (forward and backward):
`python -u tests/check_rtx5090.py --deltanet 1 --seed 999`. Then one full seed with `RTX_DELTANET=1`;
W&B name `deltanet-l5-1gpu-seed-<seed>`. Only if it beats 3.2949: seeds 1000-1003.

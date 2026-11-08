# Block AttnRes as ported (architecture reference)

Ported unchanged from `experiment/attention-residuals` (Experiment 1, commit `7a47c05`), whose five
seeds ran at 9655fc3. On `experiment/attnres-last-init` only block mode exists and it always runs; the
one change, the last-source init, and the run commands are in `docs/attnres-last-init.md`.

Block mode replaces residual-stream routing with softmax attention over completed block sums and the
current partial block. It is not stacked on the old MUDD residual paths. `RTX_ATTNRES_INJECT=1`
(default) keeps the x0/n-gram injections and layer-6 skip; `0` drops them too.

Block AttnRes follows the [official paper/pseudocode](https://github.com/MoonshotAI/Attention-Residuals).
Each active attention/MLP site has one learned query; a final readout has its own. The key RMSNorm
weight is folded into the query (logits only see their product). Queries start at zero; values remain
unnormalized. Each completed block's per-token 1/RMS is computed once, when the block closes.
Scores/softmax use FP32, weighted values use the activation dtype. States are local to one forward.
Default block size: six active serial sublayers. There are 17 sites (7 attention, 10 MLP),
forming blocks of 6/6/5 plus the initial embedding source. Layer 8's parallel MLP outputs are
summed as one site. Routing is per token, never across sequence positions.

Always replaced in block mode: residual scaling (resid lambdas), shared late attention input
(layers 8 and 10 read norm(cache[7]) in the baseline), last-layer MUDD residual/value recombination
and post-loop MUDD mix.
With `RTX_ATTNRES_INJECT=1` (default) the gated x0 / n-gram injections and the layer-6 skip are kept:
each is added to the open block like a sublayer output without counting as a site, and the skip
re-adds the summed stream after layer 3 (the baseline's `cache[3]`). Layer 10's `mu[10] * cache[0]`
and `mu[11] * n-gram` terms become `attnres_last_inject`: two learned scalars at MUDD's initial values
(0, 0.05), not per-token. With `0` all of these are dropped (the original arm).

**Isolation is partial, not "only routing".** Even with injections on, this arm still differs from the
baseline in: residual scaling, the shared late attention input, per-token vs. scalar layer-10
injection lambdas, and a new layer-10 value-gate parameterization. Attribute results to "block AttnRes
as adapted here", not to depth routing alone.
Retained: initial ngram injection, attention head/XSA gates, value embeddings, output gains,
all active attention/MLP blocks, FP8 kernels, windows and loss. The last layer's value-embedding gate
is a new parameterization, `2 + linear(gate_in)` per head (`attnres_ve_gate`, zero init = MUDD's initial
factor 2); it is not MUDD's original per-token gate restored.
Layer 4's depth mix is computed once for the post gate and the MLP unless an injection changes it.
This is an architectural adaptation, not a bitwise-equivalent optimization or exact paper reproduction.
Unused original parameters remain allocated to preserve common initialization and optimizer bookkeeping;
parameters without gradients are skipped. The record's 8-GPU path (no PORTABLE) keeps baseline routing.

New queries use existing replicated Adam: base LR 0.008 × 0.25, betas (0.9, 0.99), no weight
decay; `attnres_ve_gate` copies `ve_gate_bank`'s settings. Existing two-step accumulation/schedule.
They remain FP32; existing tail averages still apply to the original parameter groups only.
No updates/layers are reduced. Keep 1194 updates and validation packing fixed.

Experiment 1 result (five seeds, unpaired, at 9655fc3): loss 3.330801 ± 0.002697, time 948.471 ± 2.975 s,
i.e. +0.035944 loss and +10.08% time against the five-seed baseline. Not promoted.

# Layer smear: the input smear at every layer

Branch `experiment/layer-smear`, from `baseline` (5be1ba0). The model gains one parameter; the data,
schedule, loss, optimizer settings of every existing parameter and validation are unchanged.

## What changes

The record's smear mixes a gated share of the previous token's embedding into each token's, once, at the
input. Here every layer also starts with

    x_t <- x_t + w_i * x_(t-1)        (model/layers.py token_shift; the first position sees zeros)

on the residual stream, where `w_i` is a learned 768-vector for layer `i` (`layer_smear`, shape [11, 768]).
It starts at zero, so the first forward is bitwise the baseline's. It trains with the other small
replicated parameters (Adam, betas 0.9/0.99, lr_mul 1, no weight decay). Like the input smear it does not
stop at document boundaries; only each document's first position sees the previous document.

## Why

- The smear record (2025-09-18) found attention heads "consistently attending to the prior token" and
  added the smear because attention is an inefficient way to do that. A later record (2025-09-21) cites
  the smear as one reason it could drop the first attention layer. Nothing extended it inside the network.
- ANVIL2 keeps attention on 7 of 11 layers, most of them narrow; its ablation prices that package at
  6.9 millinats of loss. Cheap local mixing could take over local work and win some of it back.
- RWKV's token shift, Primer's depthwise convolution after Q/K/V, Canon layers (2025) and the short
  convolution of Mamba / Gated DeltaNet layers all add the same kind of local mixing.

It aims at the loss. On 8xH100 a lower loss becomes speed through fewer steps (~164 ms per millinat, the
record's rate). Its cost is one pointwise operation per layer; if the compiler does not fuse it, it is a
few percent of an 8xH100 step, and the loss gain must pay for that.

## Implementation note: an opaque op

Written as plain ops inside the compiled model, the shift gave `layer_smear` a non-finite gradient at
the first backward on the RTX 5090 (smoke test, 2026-11-01), while `tests/debug_layer_smear.py` showed
every incoming gradient and residual finite and small (largest |g| 0.018, largest weight gradient ~8):
the compiled code was at fault, not the values. `token_shift` is now a custom op with a hand-written
backward (the debug run with such an op passed the smoke test). The compiler cannot fuse it, so the
time this screen measures overstates what a fused kernel would cost; if the loss wins, that comes next.

## Screen

Seed 999, a pair in one allocation: the `baseline` branch, then this branch. Loss: a win needs more than
0.002 below the pair's baseline run (and the earlier seed-999 runs 3.294980 / 3.295181 / 3.295634).
Time: the difference to the pair's baseline run is the cost. The run log prints
`LAYER_SMEAR per_channel layers=11`; W&B runs are `layer-smear-1gpu-seed-<seed>`.

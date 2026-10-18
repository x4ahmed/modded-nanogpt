"""Single-GPU (RTX 5090 / SM120) port of the track-1 record (ANVIL2, #92).

The record assumes 8xH100: one microbatch per rank per step, an 84.6M-row n-gram table sharded across
ranks, a Hopper-only FA3 build, and CUDA graphs of the whole step. This package holds what replaces
those pieces when the trainer runs as ONE process (train_gpt_rtx.py). The record's own modules keep
their code paths; the few places that branch on PORTABLE say so.

What changes, and what it costs:
  - 8 ranks -> K sequential microbatches of a fixed size per step (config.MICROBATCH_TOKENS). Gradients
    are summed per microbatch, then scaled by 1/VIRTUAL_WORLD (= 8): exactly the record's AVG over 8
    ranks, whatever K is.
  - FA3 varlen -> FlexAttention with document-causal sliding-window block masks built eagerly.
  - The n-gram table lives whole on the one GPU, at 1/NGRAM_DIVISOR of the record's rows (it does not
    fit at full size: 130 GB). No row cache, no pulls: the forward reads the table directly.
  - value_embeds: dense Adam on its event steps (one rank owns every row).
  - No step / optimizer / fp8-refresh CUDA graphs: everything runs eagerly (compiled model).
"""

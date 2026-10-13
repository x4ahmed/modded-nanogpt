CPU checks for the single-GPU port (no GPU needed): `python tests/rtx/test_<name>.py .` from the repo root.
- test_attn.py: the FlexAttention masks reproduce FA3's varlen causal-window semantics (unequal QK/V widths, paired heads)
- test_grads.py: microbatch gradient accumulation and the 1/8 rank-average scale
- test_data.py: microbatch slicing and validation chunks (synthetic shards)

"""Minimal AttnRes: one depth-attention readout in place of the post-loop MUDD mix.

Adapted from https://github.com/MoonshotAI/Attention-Residuals (paper/pseudocode): softmax over depth
with RMS-normalized keys and unnormalized values, per token, no mixing across positions. Every layer
before the readout is the unchanged baseline. One learned query, plus one learned logit bias per
source; the bias starts with the final stream x at weight 0.9, near the baseline mix's init, which
is dominated by x (a deviation from the paper's uniform start, see docs/minimal-attnres.md).
"""
import math

import torch

# Initial softmax weight of the final stream x (source 0); the other sources share the rest.
X_INIT_WEIGHT = 0.9


def initial_bias(num_sources: int) -> torch.Tensor:
    """Logit biases giving x weight X_INIT_WEIGHT and the other sources equal shares (query = 0)."""
    bias = torch.zeros(num_sources)
    bias[0] = math.log(X_INIT_WEIGHT * (num_sources - 1) / (1 - X_INIT_WEIGHT))
    return bias


def depth_readout(sources, query, bias, eps=1e-6):
    """sum_k softmax_k(<v_k / rms(v_k), query> + bias_k) * v_k, per token; sources are (B, T, D)."""
    logits = torch.stack([
        (v.float() @ query) * torch.rsqrt(v.float().square().mean(-1) + eps) for v in sources
    ]) + bias.float().view(-1, *([1] * (sources[0].dim() - 1)))
    weights = logits.softmax(dim=0)
    out = sources[0] * weights[0].to(sources[0].dtype).unsqueeze(-1)
    for k in range(1, len(sources)):
        out = out + sources[k] * weights[k].to(sources[k].dtype).unsqueeze(-1)
    return out

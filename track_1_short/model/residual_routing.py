"""Depth attention over completed block sums and the current partial sum.

Adapted from https://github.com/MoonshotAI/Attention-Residuals (paper/pseudocode).
Keys are RMS-normalized; values are NOT normalized. Softmax is over depth only.
No token mixing, detached history, or persistent state across forwards.

A completed block is stored as (value, inv_rms): its per-token 1/RMS is computed once, when the
block closes, and reused by every later site. The key's RMSNorm weight is folded into the query
(the logit only ever sees their elementwise product), so each site learns one query vector.

Last-source init (experiment/attnres-last-init): each site also learns one scalar bias on its newest
source (the open block, else the newest completed block). With n sources the newest logit gains
bias + log(n - 1), so at zero query the newest source gets weight sigmoid(bias) and the other n - 1
share the rest equally, whatever n is. The bias starts at logit(LAST_SOURCE_INIT_WEIGHT). The paper's
uniform start (weight 1/n) is bias = -log(n - 1); the zero queries are unchanged.
"""
import math

import torch

# Initial depth-softmax weight of each site's newest source (zero query); the others share the rest.
LAST_SOURCE_INIT_WEIGHT = 0.9


def last_bias_init(weight=LAST_SOURCE_INIT_WEIGHT):
    """The last-source bias at which the newest source starts with `weight` of the depth softmax."""
    return math.log(weight / (1 - weight))


def inv_rms(source, eps=1e-6):
    """Per-token 1/RMS of a depth source, in fp32: the key normalization without a stored key."""
    return torch.rsqrt(source.float().square().mean(-1) + eps)


def source(value):
    """A completed depth source, (value, inv_rms)."""
    return value, inv_rms(value)


def block_mix(blocks, partial, query, last_bias=None):
    sources = blocks if partial is None else (*blocks, source(partial))
    # logit = <value / rms(value), query>, as one fp32 dot per source and token.
    logits = [(value.float() @ query) * scale for value, scale in sources]
    if last_bias is not None and len(sources) > 1:
        # A single source always has weight 1, so its site's bias is unused there.
        logits[-1] = logits[-1] + (last_bias.float() + math.log(len(sources) - 1))
    weights = torch.stack(logits, dim=0).softmax(dim=0)
    result = sources[0][0] * weights[0].to(sources[0][0].dtype).unsqueeze(-1)
    for j in range(1, len(sources)):
        value = sources[j][0]
        result = result + value * weights[j].to(value.dtype).unsqueeze(-1)
    return result


def append_output(blocks, partial, output, completed, block_size):
    """Accumulate only sublayer outputs, never the routed input (no double counting)."""
    partial = output if partial is None else partial + output
    completed += 1
    if completed % block_size == 0:
        return (*blocks, source(partial)), None, completed
    return blocks, partial, completed


def add_to_partial(partial, extra):
    """Add a non-sublayer term (an x0 / n-gram injection, the layer-6 skip) to the open block.

    It does not count as a site, so block boundaries stay at the same sublayers."""
    if extra is None:
        return partial
    return extra if partial is None else partial + extra


def stream_total(blocks, partial):
    """The plain residual stream: every source summed (the baseline's x at this point)."""
    total = blocks[0][0]
    for value, _ in blocks[1:]:
        total = total + value
    return total if partial is None else total + partial

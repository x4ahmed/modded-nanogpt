"""Bigram output prior: a count-based, smoothed bigram model added to the lm_head logits.

What: for each previous token a, the K most frequent continuations b carry a fixed value
    val(a, b) = log p(b | a) - log p(b),   p(b | a) = (c(a, b) + kappa * p(b)) / (c(a) + kappa)
(Dirichlet-smoothed bigram over a unigram backoff, p(b) add-one smoothed), and the training and
validation losses add  beta * val(input_token, b)  to the raw logit of class b before the softcap.
Every other class gets 0, i.e. "no evidence beyond the unigram". beta is one learned scalar (init 1).
This is a product-of-experts combination with the transformer, not probability interpolation;
the softcap compresses large values, so beta's learned value is part of the result.

Counts come from the first RTX_BIGRAM_TOKENS tokens of the training shards (sorted order, the same
stream the run then trains on), built on the GPU inside the timed region. Validation data is never read.

Storage: idx[a, k] (int32 class id, 0 = empty slot) and val[a, k] (fp32). Class 0 is reserved as the
empty marker, so it never receives a prior; invalid slots point at column 0 with value 0, which makes
the scatter's duplicate writes to column 0 identical (deterministic).

The input-side n-gram table (ngram_table.py) feeds hashed bigram/trigram embeddings into the residual
stream; this prior acts on the output instead.
"""
import torch
from torch import Tensor


def build_bigram_prior(tokens: Tensor, vocab_size: int, k: int, kappa: float) -> tuple[Tensor, Tensor]:
    """(idx [vocab_size, k] int32, val [vocab_size, k] fp32) from a 1-D token stream (any int dtype)."""
    tokens = tokens.long()
    prev, nxt = tokens[:-1], tokens[1:]
    n = prev.numel()
    device = tokens.device
    count_a = torch.bincount(prev, minlength=vocab_size).double()
    p_b = (torch.bincount(nxt, minlength=vocab_size).double() + 1) / (n + vocab_size)

    keys, counts = torch.unique(prev * vocab_size + nxt, return_counts=True)
    a, b = keys // vocab_size, keys % vocab_size
    keep = b != 0  # class 0 marks empty slots
    a, b, counts = a[keep], b[keep], counts[keep].double()
    val = (torch.log(counts + kappa * p_b[b]) - torch.log(count_a[a] + kappa) - torch.log(p_b[b])).float()

    # Group by a, most frequent continuation first; rank within the group.
    order = torch.argsort(-counts, stable=True)
    order = order[torch.argsort(a[order], stable=True)]
    a, b, val = a[order], b[order], val[order]
    rank = torch.arange(a.numel(), device=device) - torch.searchsorted(a, a, right=False)
    top = rank < k

    idx_table = torch.zeros(vocab_size, k, dtype=torch.int32, device=device)
    val_table = torch.zeros(vocab_size, k, dtype=torch.float32, device=device)
    idx_table[a[top], rank[top]] = b[top].int()
    val_table[a[top], rank[top]] = val[top]
    return idx_table, val_table


def prior_columns(idx: Tensor, val: Tensor, prev: Tensor, col_of_class: Tensor | None = None):
    """Per-row (cols [T, k] int64, vals [T, k] fp32) for rows whose previous token is `prev` [T].

    col_of_class maps a class to its logit column (the sampled-softmax candidate position, -1 if absent);
    None means the full vocabulary. Slots without a usable column point at column 0 with value 0."""
    classes = idx[prev.long()].long()
    vals = val[prev.long()]
    cols = classes if col_of_class is None else col_of_class[classes].long()
    valid = (classes > 0) & (cols > 0)
    return torch.where(valid, cols, 0), torch.where(valid, vals, 0.0)


def add_prior_(logits: Tensor, cols: Tensor, vals: Tensor, beta: Tensor) -> Tensor:
    """logits[r, cols[r, j]] += beta * vals[r, j], in place. fp8 logits are updated through a byte view
    (gather, add in fp32, round once, scatter); float logits directly."""
    delta = beta.float() * vals
    if logits.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        raw = logits.view(torch.uint8)
        current = raw.gather(1, cols).view(logits.dtype).float()
        raw.scatter_(1, cols, (current + delta).to(logits.dtype).view(torch.uint8))
    else:
        current = logits.gather(1, cols).float()
        logits.scatter_(1, cols, (current + delta).to(logits.dtype))
    return logits


def beta_grad(grad_input: Tensor, cols: Tensor, vals: Tensor, grad_s: float) -> Tensor:
    """d(sum of losses)/d beta from the CE kernel's e5m2 logit gradient (true gradient = grad_input * grad_s)."""
    g = grad_input.view(torch.uint8).gather(1, cols).view(grad_input.dtype).float()
    return (g * vals).sum() * grad_s


def read_train_tokens(pattern: str, n_tokens: int, device) -> Tensor:
    """The first n_tokens of the training shards, in the loader's (sorted) order, on `device`."""
    import glob
    from pathlib import Path

    from track_1_short.data import _load_data_shard

    parts, have = [], 0
    for file in sorted(glob.glob(pattern)):
        shard = _load_data_shard(Path(file))[: n_tokens - have]
        parts.append(shard.to(device, non_blocking=True))
        have += shard.numel()
        if have >= n_tokens:
            break
    assert have >= 2, f"no training tokens under {pattern}"
    return torch.cat(parts)

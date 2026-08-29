"""Exact document-causal sliding-window BlockMasks for the portable backend."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


BLOCK_SIZE = 128


def _dense_to_ordered(dense_mask: Tensor) -> tuple[Tensor, Tensor]:
    num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
    indices = dense_mask.argsort(dim=-1, descending=True, stable=True).to(torch.int32)
    return num_blocks[None, None].contiguous(), indices[None, None].contiguous()


def _build_document_block_mask(document_ids: Tensor, window_tokens: int) -> BlockMask:
    """Build the exact FA3 mask: same document, causal, q-kv <= window."""
    assert document_ids.ndim == 1
    assert document_ids.numel() % BLOCK_SIZE == 0
    assert window_tokens >= 0 and window_tokens % BLOCK_SIZE == 0

    num_tokens = document_ids.numel()
    num_blocks = num_tokens // BLOCK_SIZE
    block = torch.arange(num_blocks, dtype=torch.int32, device=document_ids.device)
    q0 = (block * BLOCK_SIZE)[:, None]
    q1 = q0 + BLOCK_SIZE - 1
    k0 = (block * BLOCK_SIZE)[None, :]
    k1 = k0 + BLOCK_SIZE - 1

    docs_low = document_ids.view(-1, BLOCK_SIZE)[:, 0].contiguous()
    docs_high = document_ids.view(-1, BLOCK_SIZE)[:, -1].contiguous()
    document_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
    document_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)

    causal_any = q1 >= k0
    causal_all = q0 > k1
    window_any = q0 - k1 <= window_tokens
    window_all = q1 - k0 <= window_tokens
    block_any = causal_any & window_any & document_any
    block_all = causal_all & window_all & document_all

    partial_num, partial_indices = _dense_to_ordered(block_any & ~block_all)
    full_num, full_indices = _dense_to_ordered(block_all)

    def document_causal_window(_batch, _head, q_idx, kv_idx):
        return (
            (q_idx >= kv_idx)
            & (q_idx - kv_idx <= window_tokens)
            & (document_ids[q_idx] == document_ids[kv_idx])
        )

    return BlockMask.from_kv_blocks(
        partial_num,
        partial_indices,
        full_num,
        full_indices,
        BLOCK_SIZE=BLOCK_SIZE,
        mask_mod=document_causal_window,
        seq_lengths=(num_tokens, num_tokens),
    )


def create_document_block_masks(
    seqlens: Tensor,
    num_tokens: int,
    window_tokens: tuple[int, ...],
    *,
    paired: bool,
) -> tuple[BlockMask, ...]:
    """Build requested masks; paired heads double positions, not window widths."""
    # seqlens stores end-exclusive cumulative document boundaries. Build IDs by
    # incrementing at each boundary; padded boundaries equal num_tokens and fall
    # in the extra element that is discarded below.
    document_starts = torch.zeros(num_tokens + 1, dtype=torch.int32, device=seqlens.device)
    boundaries = seqlens[1:].to(torch.int64)
    document_starts.scatter_add_(
        0,
        boundaries,
        torch.ones_like(boundaries, dtype=document_starts.dtype),
    )
    document_ids = document_starts.cumsum(0)[:num_tokens]
    if paired:
        document_ids = document_ids.repeat_interleave(2)
    return tuple(
        _build_document_block_mask(document_ids, window)
        for window in window_tokens
    )

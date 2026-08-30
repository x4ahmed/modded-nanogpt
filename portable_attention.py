"""Exact document-causal sliding-window BlockMasks for the portable backend."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


BLOCK_SIZE = 128


def _dense_to_ordered(dense_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Selected columns first in ascending order, unselected after, without a sort.

    This is exactly `argsort(dim=-1, descending=True, stable=True)` on a boolean mask,
    computed with cumsum and scatter instead. Inductor lowers torch.argsort to a Triton
    bitonic sort that stages values and indices in shared memory; at the PORTABLE eval
    sequence length the mask is 512x512 and the kernel requested 180,248 B against
    SM120's 101,376 B limit, failing with "No valid triton configs". Measured on an
    RTX 5090, 2026-08-30. cumsum and scatter carry no such block-size pressure.
    """
    num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
    selected = dense_mask.to(torch.int64)
    # Rank among selected entries, and among unselected entries, both in column order.
    selected_rank = selected.cumsum(dim=-1) - 1
    unselected_rank = (1 - selected).cumsum(dim=-1) - 1
    slot = torch.where(
        dense_mask,
        selected_rank,
        num_blocks.to(torch.int64)[..., None] + unselected_rank,
    )
    columns = torch.arange(
        dense_mask.shape[-1], device=dense_mask.device, dtype=torch.int32
    ).expand_as(dense_mask)
    indices = torch.empty_like(columns)
    indices.scatter_(-1, slot, columns)
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

    block_partial = block_any & ~block_all
    partial_num, partial_indices = _dense_to_ordered(block_partial)
    full_num, full_indices = _dense_to_ordered(block_all)
    # Build the q-side tables here rather than through BlockMask.from_kv_blocks.
    # That helper derives them with _transpose_ordered, which calls PyTorch's own
    # _dense_to_ordered and its torch.argsort. Inductor lowers that argsort to a
    # Triton sort needing 180,248 B of shared memory against SM120's 101,376 B, and
    # because the call lives inside PyTorch it survives every change on our side.
    # Transposing the dense matrices we already have costs nothing and is exact.
    q_num, q_indices = _dense_to_ordered(block_partial.transpose(-2, -1))
    full_q_num, full_q_indices = _dense_to_ordered(block_all.transpose(-2, -1))

    def document_causal_window(_batch, _head, q_idx, kv_idx):
        return (
            (q_idx >= kv_idx)
            & (q_idx - kv_idx <= window_tokens)
            & (document_ids[q_idx] == document_ids[kv_idx])
        )

    return BlockMask(
        seq_lengths=(num_tokens, num_tokens),
        kv_num_blocks=partial_num,
        kv_indices=partial_indices,
        full_kv_num_blocks=full_num,
        full_kv_indices=full_indices,
        q_num_blocks=q_num,
        q_indices=q_indices,
        full_q_num_blocks=full_q_num,
        full_q_indices=full_q_indices,
        BLOCK_SIZE=(BLOCK_SIZE, BLOCK_SIZE),
        mask_mod=document_causal_window,
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


# Inductor's default FlexAttention template at QK_HEAD_DIM=128 picks BLOCK_M=128,
# BLOCK_N=64, num_stages=2 and requests 180,248 B of shared memory per block. SM120
# (RTX 5090) caps at 101,376 B, so compilation dies with
#   "No valid triton configs. OutOfResources: out of resource: shared memory,
#    Required: 180248, Hardware limit: 101376"
# Measured on an RTX 5090, 2026-08-30. This is a device limit, not a fallback: the
# math is unchanged, only the tile sizes the template iterates with.
#
# The template asserts SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0 and
# SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0, and both sparse sizes equal BLOCK_SIZE above,
# so every value here must divide 128.
KERNEL_OPTIONS = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "num_stages": 1,
    # Backward carries dq/dk/dv accumulators on top of the forward tiles, so it needs
    # more headroom than the forward pass.
    "BLOCK_M1": 32,
    "BLOCK_N1": 64,
    "BLOCK_M2": 64,
    "BLOCK_N2": 32,
}

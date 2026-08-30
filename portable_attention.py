"""Exact document-causal sliding-window BlockMasks for the portable backend."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


BLOCK_SIZE = 128


def _dense_to_ordered(dense_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Selected columns first in ascending order, unselected after.

    Plain argsort. These masks are built in eager mode, outside the compiled region --
    see build_window_masks -- where argsort is a CUB radix sort with no Triton
    block-size pressure. Building them inside torch.compile is what forced two
    successive rewrites here: argsort lowered to a Triton bitonic sort that wanted
    180,248 B of shared memory against SM120's 101,376 B, and the cumsum/scatter
    replacement hit an inductor codegen bug on the scatter. Keep this eager.
    """
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

    # from_kv_blocks derives the q-side tables with its own argsort. That is fine here
    # because build_window_masks runs eagerly; it was not fine inside torch.compile.
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


def build_window_masks(
    seqlens: Tensor, num_tokens: int, ws_short: int, ws_long: int
) -> tuple[BlockMask, BlockMask, BlockMask]:
    """The three masks a forward pass needs: (normal short, normal long, paired short).

    MUST be called outside the compiled region. BlockMask construction is tensor work
    that torch.compile has no reason to trace, and tracing it is what produced both the
    Triton shared-memory failure and the inductor scatter codegen bug on SM120. Built
    eagerly and passed in, the compiled graph sees a finished BlockMask -- the pattern
    FlexAttention documents.
    """
    normal_short, normal_long = create_document_block_masks(
        seqlens, num_tokens, (ws_short, ws_long), paired=False
    )
    (paired_short,) = create_document_block_masks(
        seqlens, num_tokens, (ws_short,), paired=True
    )
    return normal_short, normal_long, paired_short


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

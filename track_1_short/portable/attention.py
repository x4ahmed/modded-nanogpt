"""FlexAttention in place of the record's Hopper-only FA3 build: exact document-causal window masks.

The record calls a patched FA3 (sm_90a only) as
    flash_attn_varlen_func(q, k, v, cu_seqlens, causal=True, window_size=(window, 0))
i.e. query t attends to keys s of the same document (segment) with t - window <= s <= t. The same
mask as a FlexAttention BlockMask: same segment, causal, q - kv <= window. Paired-head layers double
the token positions (and the segment boundaries) but keep the window, as the record does.

The masks MUST be built outside the compiled model and passed in (build_window_masks): building a
BlockMask inside torch.compile traced its argsort into a Triton sort that overflowed SM120's shared
memory, and the scatter replacement hit an inductor codegen bug (both seen in rtx-local-v1).

Unequal query/key and value head widths (the record's 64/128 mixed widths) are native to FlexAttention.

Carried over from rtx-local-v1 (portable_attention.py), where it ran on an RTX 5090.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

BLOCK_SIZE = 128


def _dense_to_ordered(dense_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Selected columns first in ascending order, unselected after. Eager only (see the module doc)."""
    num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
    indices = dense_mask.argsort(dim=-1, descending=True, stable=True).to(torch.int32)
    return num_blocks[None, None].contiguous(), indices[None, None].contiguous()


def _build_document_block_mask(document_ids: Tensor, window_tokens: int) -> BlockMask:
    """Same document, causal, q - kv <= window."""
    assert document_ids.ndim == 1
    assert document_ids.numel() % BLOCK_SIZE == 0
    assert window_tokens >= 0

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


def document_ids_from_seqlens(seqlens: Tensor, num_tokens: int) -> Tensor:
    """[num_tokens] segment id of every token, from the packed cumulative boundaries the loader makes
    (cu_seqlens: starts with 0, padded with num_tokens). Padding boundaries fall in a discarded slot."""
    document_starts = torch.zeros(num_tokens + 1, dtype=torch.int32, device=seqlens.device)
    boundaries = seqlens[1:].to(torch.int64).clamp(max=num_tokens)
    document_starts.scatter_add_(0, boundaries, torch.ones_like(boundaries, dtype=document_starts.dtype))
    return document_starts.cumsum(0)[:num_tokens].to(torch.int32)


@torch.no_grad()
def build_window_masks(seqlens: Tensor, num_tokens: int, ws_short: int, ws_long: int) -> tuple[BlockMask, ...]:
    """(normal short, normal long, paired short): every mask one forward needs. Eager, outside compile.
    Windows are in tokens."""
    document_ids = document_ids_from_seqlens(seqlens, num_tokens)
    normal_short = _build_document_block_mask(document_ids, ws_short)
    normal_long = _build_document_block_mask(document_ids, ws_long)
    paired_short = _build_document_block_mask(document_ids.repeat_interleave(2), ws_short)
    return normal_short, normal_long, paired_short


# Inductor's default FlexAttention tiles at head dim 128 ask for 180,248 B of shared memory per block;
# SM120 (RTX 5090) has 101,376 B. These tiles fit (measured on an RTX 5090 in rtx-local-v1); the math is
# unchanged. Every value divides the masks' BLOCK_SIZE of 128, as the template requires. The backward
# carries dq/dk/dv accumulators, so its tiles are smaller.
KERNEL_OPTIONS = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "num_stages": 1,
    "BLOCK_M1": 32,
    "BLOCK_N1": 64,
    "BLOCK_M2": 64,
    "BLOCK_N2": 32,
}

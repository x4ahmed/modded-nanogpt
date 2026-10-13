"""Data for one process playing the record's ranks: K fixed-size microbatches per step, and the
validation stream cut into the per-rank chunks the record evaluates.

Training: the record gives each of 8 ranks one BOS-aligned slice of batch_size / 8 tokens per step
(data.py Shard.next_batch). Here a step is K = batch_size / MICROBATCH_TOKENS slices of MICROBATCH_TOKENS
tokens each, cut by the same Shard.next_batch (with K in place of the world size), so every step trains
on the same number of tokens as the record's and every microbatch is one shape for the compiled model.
The token stream differs slightly from the record's: more, shorter slices truncate a few more documents
at slice ends (the same trade rtx-local-v1 made).

Validation: the record reads val_batch_size tokens per val step, rank r taking the r-th contiguous
chunk of val_batch_size / 8. With VAL_SEQ_LEN = val_batch_size / 8 the chunks here are exactly those,
one forward each, so the validation loss is the record's to the bit of the data.
"""
import glob
from pathlib import Path

import torch

from track_1_short.config import VIRTUAL_SEQ_CAP
from track_1_short.data import BOS_ID, Batch, Shard, _load_data_shard, cu_seqlens_rows, split_attention_segments
from track_1_short.model.layers import next_multiple_of_n
from track_1_short.ngram_table import ngram_row_ids
from track_1_short.perf.pinned_batches import PinnedBatchStaging


def _files(pattern: str) -> list[Path]:
    files = [Path(f) for f in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return files


def microbatch_cu_seqlens_rows(microbatch_tokens: int) -> int:
    """Rows of a training microbatch's cu_seqlens table. The record sizes it per batch shape from its own
    slicing (data.TRAIN_MAX_NUM_DOCS: 64 rows at 16384 tokens, seen only with stage 0's data); the slices
    here differ, so allow one segment per 64 tokens. Only the mask builder reads the table."""
    return max(cu_seqlens_rows(microbatch_tokens), next_multiple_of_n(microbatch_tokens // 64, n=128))


def _make_batch(inputs: torch.Tensor, targets: torch.Tensor, cum_lengths: torch.Tensor, num_tokens: int,
                max_num_docs: int, staging: PinnedBatchStaging) -> Batch:
    """The record's batch assembly (data.distributed_data_generator), from one slice's tokens."""
    assert len(cum_lengths) < max_num_docs, f"{len(cum_lengths)} segments overflow the {max_num_docs}-row cu_seqlens table"
    _cum_lengths = torch.full((max_num_docs,), num_tokens)
    _cum_lengths[0] = 0
    _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths
    _inputs = inputs.to(dtype=torch.int32)
    _targets = targets.to(dtype=torch.int64)
    _cum_lengths = _cum_lengths.to(dtype=torch.int32)
    _ngram_ids = ngram_row_ids(_inputs)
    inputs_d, targets_d, cum_seqlens_d, ngram_ids_d = staging.upload(_inputs, _targets, _cum_lengths, _ngram_ids)
    return Batch(inputs=inputs_d, targets=targets_d, cum_seqlens=cum_seqlens_d, ngram_ids=ngram_ids_d,
                 ngram_ids_cpu=_ngram_ids.numpy(), targets_cpu=_targets, inputs_cpu=_inputs.numpy())


class MicrobatchLoader:
    """Training batches: next_step(K, max_seq_len) -> K microbatches of `microbatch_tokens` each."""

    def __init__(self, pattern: str, microbatch_tokens: int, staging: PinnedBatchStaging):
        self.microbatch_tokens = microbatch_tokens
        self.staging = staging
        self.file_iter = iter(_files(pattern))
        self.shard = Shard(_load_data_shard(next(self.file_iter)))
        self.next_shard_getter = self._preload()

    def _preload(self):
        try:
            return Shard.load_async(next(self.file_iter))
        except StopIteration:
            return None

    def next_step(self, num_microbatches: int, max_seq_len: int) -> list[Batch]:
        mb = self.microbatch_tokens
        while True:
            # Shard.next_batch cuts one slice per "rank": here one per microbatch.
            self.shard.world_size = num_microbatches
            try:
                starts, ends = self.shard.next_batch(mb, max_seq_len)
                break
            except StopIteration:
                assert self.next_shard_getter is not None, "out of training data"
                self.shard = self.next_shard_getter()
                self.next_shard_getter = self._preload()
        tokens = self.shard.tokens
        batches = []
        for seq_starts, seq_ends in zip(starts, ends):
            start_idxs, end_idxs = torch.tensor(seq_starts), torch.tensor(seq_ends)
            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            end_idxs[-1] -= 1  # the last document was cut one short for the targets' offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)
            if max_seq_len > VIRTUAL_SEQ_CAP:
                cum_lengths = split_attention_segments(cum_lengths, VIRTUAL_SEQ_CAP)
            batches.append(_make_batch(buf[:-1], buf[1:], cum_lengths, mb, microbatch_cu_seqlens_rows(mb), self.staging))
        return batches


def validation_batches(pattern: str, val_tokens: int, seq_len: int, staging: PinnedBatchStaging):
    """The validation stream in chunks of `seq_len` tokens (generator: one chunk uploaded at a time)."""
    assert val_tokens % seq_len == 0
    file_iter = iter(_files(pattern))
    tokens = _load_data_shard(next(file_iter))
    pos = 0
    for _ in range(val_tokens // seq_len):
        if pos + seq_len + 1 >= len(tokens):  # should not occur for val data
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos:pos + seq_len + 1]
        inputs = buf[:-1]
        cum_lengths = torch.nonzero(inputs == BOS_ID)[:, 0]
        pos += seq_len
        # The record's own table size for its per-rank validation chunk.
        yield _make_batch(inputs, buf[1:], cum_lengths, seq_len, cu_seqlens_rows(seq_len), staging)

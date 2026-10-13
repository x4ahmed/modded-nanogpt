# rtx-local-v2: record #92 (ANVIL2) on one RTX 5090

Branch `rtx-local-v2` is upstream `master` at 4ea6b93 (record #92, 39.9 s on 8xH100) plus a single-GPU
port. `rtx-local-v1` (record #89 on one 5090) is untouched.

## Run it

```bash
python data/cached_fineweb10B.py 9          # once: first train shards + the val shard
RTX_PREFLIGHT=1 bash run_rtx.sh             # 1) memory preflight: ~5-10 min, prints PASS/FAIL and exits
bash run_rtx.sh                             # 2) the full run
RTX_MAX_STEPS=50 bash run_rtx.sh            #    or a short smoke run (validates after 50 steps)
```

Stack: the same one rtx-local-v1 used on the 5090 (torch 2.10.0+cu128, CUDA_HOME pointing at a CUDA
tree with `cuda_bf16.h`). The FA3 kernel, `kernels` and huggingface-hub are not needed by this path.

The preflight runs the whole warmup -- every compiled configuration, the largest microbatch, both
softmax kinds, the table updates -- and one validation forward at full length, then prints a
`MEMORY verdict` line and `PREFLIGHT PASS|FAIL`. If it fails, lower one knob and rerun:

| knob | default | effect |
|---|---|---|
| `RTX_MICROBATCH_TOKENS` | 16384 | activation memory per microbatch; 8192 halves it (more microbatches, slower) |
| `RTX_NGRAM_DIVISOR` | 32 | n-gram table rows = 84.6M / divisor; 64 halves its 8.2 GB (table + fp16 gradient) |
| `RTX_VAL_SEQ_LEN` | 262144 | tokens per validation forward; 65536 if the eval forward is the peak (changes the metric slightly) |
| `RTX_EVAL_CE_SLAB_ROWS` | 4096 | rows of fp32 validation logits per slab (0.8 GB at 4096) |

## What is the record, unchanged

Model (11 layers, mixed-width attention, MUDD, XSA, value embeddings, hashed n-gram embedding,
prefix-token + multi-token losses), ANVIL + Adam with every hyperparameter and schedule, the
5-stage batch/window/sequence schedule (1194 steps), sampled softmax, full-stack FP8 (MLP, QKV,
lm_head), the fused CE kernel, tail averaging, canonical-mask validation. Gradients are the record's
rank-averaged gradients exactly (summed microbatch gradients x 1/8, accumulated in fp32).

## What one GPU changes

| record (8xH100) | here (1x5090) | why |
|---|---|---|
| 8 ranks, one microbatch each per step | one process, K = batch / 16384 microbatches per step (8/16/24/20/8 by stage) | one GPU; same tokens per step |
| patched FA3 varlen (sm_90a binary) | FlexAttention + eagerly built document-causal window BlockMasks | FA3 build does not run on SM120 |
| 84.6M-row n-gram table sharded 8 ways (130 GB) | 2.64M rows (1/32) whole on the GPU, read directly by row id | does not fit; same hashes mod the smaller table |
| row pulls / caches / sparse all-to-alls | none (one owner) | nothing to exchange |
| n-gram Adam: lazy per-row replay over touched rows | same kernel over every row, zero grad where untouched | identical numbers, no gradient compaction buffer |
| CUDA graphs: step fwd/bwd, ANVIL tails, fp8 refresh | eager (compiled model) | per-microbatch work is large; launch overhead small |
| fused MLP tiles 128x256 | 128x128 when shared memory < 200 KB | SM120 has 99 KB per block |
| sampled-softmax sets per rank, prep thread | per microbatch, synchronous | microbatches run in sequence |
| data: 8 BOS-aligned slices per step | K slices of 16384 | a few more truncated documents per step |

Expected differences: the smaller n-gram table costs some loss (the record's own ablation: going back
to record #89's 377K-row bigram table costs ~36 millinats; 2.64M rows sits in between -- measure it).
The final loss may therefore land a little above the record's 3.2773 mean. Wall time is not
comparable to the record (one consumer GPU vs 8xH100); compare against the rtx-local-v1 #89 run
(3.2770 @ 33 min on the same 5090).

## Files

- `train_gpt_rtx.py` -- the trainer (warmup, timed loop, validation, preflight)
- `run_rtx.sh` -- launcher
- `track_1_short/portable/` -- `config.py` (knobs), `attention.py` (masks), `tables.py` (n-gram table,
  value_embeds), `sampled.py`, `data.py`, `grads.py`, `memory.py`
- Record modules touched, each behind `PORTABLE`: `model/attention.py` (FlexAttention path, lazy FA3
  load), `model/gpt.py` (masks into each layer, eval slab size), `ngram_table.py` (table size),
  `distributed.py` (world size 1); `perf/kernels/mlp.py` picks its tile by shared memory on any device.

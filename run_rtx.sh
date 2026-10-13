#!/usr/bin/env bash
# Track 1 (record #92, ANVIL2) on ONE RTX 5090 -- the rtx-local-v2 port. See docs/rtx-local-v2.md.
#   RTX_PREFLIGHT=1 bash run_rtx.sh   memory preflight: warmup + one validation forward, PASS/FAIL, exit
#   bash run_rtx.sh                   full run
# Knobs (environment): RTX_MICROBATCH_TOKENS, RTX_NGRAM_DIVISOR, RTX_VAL_SEQ_LEN, RTX_EVAL_CE_SLAB_ROWS,
# RTX_MAX_STEPS, TRAIN_SEED (track_1_short/portable/config.py).
set -euo pipefail
cd "$(dirname "$0")"
export PORTABLE=1
torchrun --standalone --nproc_per_node=1 train_gpt_rtx.py "$@"

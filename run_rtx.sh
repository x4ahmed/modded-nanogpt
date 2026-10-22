#!/usr/bin/env bash
# Track 1 (record #92, ANVIL2) on ONE RTX 5090 -- the rtx-local-v2 port.
# Verify the port first: python tests/check_rtx5090.py
#   RTX_PREFLIGHT=1 bash run_rtx.sh   memory preflight: warmup + one validation forward, PASS/FAIL, exit
#   bash run_rtx.sh                   full run
# Knobs (environment): RTX_MICROBATCH_TOKENS, RTX_NGRAM_DIVISOR, RTX_VAL_SEQ_LEN, RTX_EVAL_CE_SLAB_ROWS,
# RTX_MAX_STEPS, TRAIN_SEED (track_1_short/portable/config.py).
set -euo pipefail
cd "$(dirname "$0")"
export PORTABLE=1
# Capture the launch commit before training, never infer it at upload time.
training_commit=$(git rev-parse HEAD 2>/dev/null || true)
mkdir -p logs
console_log="logs/run-$(date -u +%Y%m%dT%H%M%SZ)-$$.console.txt"
torchrun --standalone --nproc_per_node=1 train_gpt_rtx.py "$@" 2>&1 | tee "$console_log"

# Only reached after successful training (pipefail). No W&B process runs on the clock.
if [[ "${RTX_WANDB_UPLOAD:-1}" == "1" && "${RTX_PREFLIGHT:-0}" != "1" ]]; then
    upload_python="${WANDB_UPLOAD_PYTHON:-${HOME}/venvs/wandb-upload/bin/python}"
    upload_args=("$console_log" --entity i-learn --project speedrun)
    [[ -z "${TRAIN_SEED:-}" ]] || upload_args+=(--seed "$TRAIN_SEED")
    [[ -z "$training_commit" ]] || upload_args+=(--training-commit "$training_commit")
    run_variant=baseline
    [[ "${RTX_READOUT:-mudd}" != "attnres" ]] || run_variant=readout-attnres
    upload_args+=(--name "${run_variant}-1gpu-seed-${TRAIN_SEED:-unseeded}")
    echo "Training finished. Uploading recorded metrics to i-learn/speedrun."
    if ! "$upload_python" tools/upload_wandb.py "${upload_args[@]}"; then
        echo "WARNING: training succeeded but W&B upload failed. Local results are safe: $console_log" >&2
        printf 'Retry after login/setup: %q tools/upload_wandb.py' "$upload_python" >&2
        printf ' %q' "${upload_args[@]}" >&2
        printf '\n' >&2
    fi
fi

"""Executable two-GPU gate for the RTX-local baseline.

Run through torchrun. This bootstrap stays stdlib-only and replaces each rank
with train_gpt.py's internal preflight path; it never imports the training script.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path


SEED_MAX = 2**32 - 1


def bounded_seed(value: str) -> int:
    try:
        seed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be an integer") from exc
    if not 0 <= seed <= SEED_MAX:
        raise argparse.ArgumentTypeError(f"seed must be between 0 and {SEED_MAX}")
    return seed


def nonnegative_finite(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("headroom must be a number") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("headroom must be finite and nonnegative")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", required=True, type=bounded_seed)
    parser.add_argument("--min-headroom-gib", type=nonnegative_finite, default=2.0)
    parser.add_argument("--local-rank", "--local_rank", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if os.environ.get("WORLD_SIZE") != "2" or os.environ.get("LOCAL_WORLD_SIZE") != "2":
        parser.error("run with torchrun --standalone --nproc_per_node=2")
    if os.environ.get("DISABLE_FP8"):
        parser.error("DISABLE_FP8 must be unset; the preflight does not permit fallbacks")

    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / "train_gpt.py"
    rank = os.environ.get("RANK", "unknown")
    print(
        f"[rank {rank}] PREFLIGHT_BOOTSTRAP seed={args.seed} "
        f"min_headroom_gib={args.min_headroom_gib:g}",
        flush=True,
    )
    child_env = os.environ.copy()
    child_env.update(
        PORTABLE="1",
        RTX_PREFLIGHT="1",
        _RTX_DEFER_CE_INIT="1",
        RTX_MIN_HEADROOM_GIB=str(args.min_headroom_gib),
        PYTHONHASHSEED=str(args.seed),
    )
    child_argv = [
        sys.executable,
        str(train_script),
        "--run-id",
        f"rtx-preflight-seed-{args.seed}",
        "--seed",
        str(args.seed),
    ]
    os.execve(sys.executable, child_argv, child_env)


if __name__ == "__main__":
    main()

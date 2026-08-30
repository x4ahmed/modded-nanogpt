"""Small, dependency-free helpers for the opt-in local RTX baseline."""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SEED_MIN = 0
SEED_MAX = 2**32 - 1
FULL_SCHEDULE_STEPS = 1285
NATIVE_VAL_BATCH_SIZE = 2_097_152
PORTABLE_VAL_BATCH_SIZE = 524_288
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    portable: bool
    run_id: str
    seed: int | None
    max_steps: int | None
    python_hash_seed: str | None


@dataclass(frozen=True, slots=True)
class RunPlan:
    optimizer_steps: int
    full_schedule_steps: int

    @property
    def complete_schedule(self) -> bool:
        return self.optimizer_steps == self.full_schedule_steps

    def iterations(self) -> Iterable[tuple[int, bool]]:
        """Yield N optimizer updates followed by one validation-only iteration."""
        for step in range(self.optimizer_steps + 1):
            yield step, step == self.optimizer_steps


def _bounded_int(name: str, minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value, 10)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be between {minimum} and {maximum}"
            )
        return parsed

    return parse


def _validate_run_id(value: str) -> str:
    if not _RUN_ID_RE.fullmatch(value) or value in {".", ".."}:
        raise argparse.ArgumentTypeError(
            "run-id must be 1-128 path-safe characters: letters, digits, '.', '_' or '-'"
        )
    return value


def parse_runtime_config(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> RuntimeConfig:
    env = os.environ if environ is None else environ
    portable = env.get("PORTABLE") == "1"
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-id", type=_validate_run_id)
    parser.add_argument("--seed", type=_bounded_int("seed", SEED_MIN, SEED_MAX))
    parser.add_argument(
        "--max-steps",
        type=_bounded_int("max-steps", 1, FULL_SCHEDULE_STEPS),
    )
    parser.add_argument("--local-rank", "--local_rank", type=int, help=argparse.SUPPRESS)
    parsed = parser.parse_args(argv)

    if portable:
        if parsed.run_id is None:
            parser.error("--run-id is required when PORTABLE=1")
        if parsed.seed is None:
            parser.error("--seed is required when PORTABLE=1")
    elif parsed.seed is not None or parsed.max_steps is not None:
        parser.error("--seed and --max-steps require PORTABLE=1")

    # PYTHONHASHSEED is recorded for the run manifest but deliberately not enforced:
    # nothing in the training path depends on hash ordering (no shuffle, and no set or
    # dict iteration feeds RNG, data order, or parameter construction), so requiring it
    # would only add a way for an otherwise-correct run to fail at argument parsing.
    return RuntimeConfig(
        portable=portable,
        run_id=parsed.run_id or str(uuid.uuid4()),
        seed=parsed.seed,
        max_steps=parsed.max_steps,
        python_hash_seed=env.get("PYTHONHASHSEED"),
    )


def make_run_plan(full_schedule_steps: int, max_steps: int | None) -> RunPlan:
    if full_schedule_steps != FULL_SCHEDULE_STEPS:
        raise ValueError(
            f"expected the pinned {FULL_SCHEDULE_STEPS}-step schedule, got {full_schedule_steps}"
        )
    optimizer_steps = full_schedule_steps if max_steps is None else max_steps
    if not 1 <= optimizer_steps <= full_schedule_steps:
        raise ValueError("optimizer step cap is outside the full schedule")
    return RunPlan(optimizer_steps, full_schedule_steps)


# Fused ReLU-squared MLP tiles, largest first: (BLOCK_SIZE_M, BLOCK_SIZE_N, num_stages).
# The first entry is the original H100 autotuning and is selected on any device with
# room for it, so the record path is unchanged.
MLP_BLOCK_CANDIDATES = (
    (128, 256, 4),
    (128, 256, 3),
    (128, 128, 3),
    (128, 128, 2),
    (64, 128, 2),
)
# Triton's own accounting includes a little fixed overhead beyond the tiles (24 bytes
# in the measured case), and occupancy suffers at the very edge, so leave a margin.
MLP_SHARED_MEMORY_MARGIN = 0.95


def mlp_shared_memory_bytes(
    block_m: int, block_n: int, block_k: int, num_stages: int, use_fp8: bool
) -> int:
    """Shared memory the fused MLP kernel needs for one block.

    Triton pipelines num_stages - 1 input buffers and holds one output tile. Validated
    against the measured failure on an RTX 5090: the original (128, 256, 64, 4) bf16
    configuration reports 180,248 B, and this returns 180,224 -- the 24-byte remainder
    is Triton's fixed overhead, covered by MLP_SHARED_MEMORY_MARGIN.
    """
    element = 1 if use_fp8 else 2
    stage_bytes = (block_m * block_k + block_n * block_k) * element
    output_bytes = block_m * (block_n // 2) * 2  # the output tile stays bf16
    return max(num_stages - 1, 1) * stage_bytes + output_bytes


def select_mlp_block_config(
    shared_memory_limit: int, use_fp8: bool
) -> tuple[int, int, int, int]:
    """Largest candidate tiles that fit the device, as (BM, BN, BK, max_num_stages).

    Selection compares against the device's own opt-in limit rather than a device name
    or a hand-picked cutoff. An A100 (163,840 B) sits between the H100 configuration's
    181 KB requirement and a 160 KiB threshold, so a threshold would misclassify it.
    """
    block_k = 128 if use_fp8 else 64
    budget = shared_memory_limit * MLP_SHARED_MEMORY_MARGIN
    for block_m, block_n, num_stages in MLP_BLOCK_CANDIDATES:
        if mlp_shared_memory_bytes(block_m, block_n, block_k, num_stages, use_fp8) <= budget:
            return block_m, block_n, block_k, num_stages
    raise RuntimeError(
        f"no fused MLP tile configuration fits {shared_memory_limit} B of shared memory"
    )


def validation_batch_size(portable: bool) -> int:
    return PORTABLE_VAL_BATCH_SIZE if portable else NATIVE_VAL_BATCH_SIZE


def validation_sequence_length(val_batch_size: int, world_size: int, grad_accum_steps: int) -> int:
    if world_size * grad_accum_steps != 8:
        raise ValueError("validation packing expects world_size * grad_accum_steps == 8")
    return val_batch_size // (world_size * grad_accum_steps)


def seed_everything(seed: int, *, random_module, numpy_module, torch_module) -> None:
    """Apply the same explicit seed on every rank; deliberately no rank offset."""
    if not SEED_MIN <= seed <= SEED_MAX:
        raise ValueError("seed outside uint32 range")
    random_module.seed(seed)
    numpy_module.random.seed(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)


def collect_git_metadata(repo: str | Path = ".") -> dict[str, str | bool]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": "unknown"}


def format_final_summary(
    final_val_loss: float,
    training_time_ms: float,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
) -> str:
    if not math.isfinite(final_val_loss):
        raise ValueError("final validation loss must be finite")
    return (
        f"final_val_loss={final_val_loss:.6f} "
        f"training_time_ms={training_time_ms:.0f} "
        f"peak_memory_allocated_bytes={peak_allocated_bytes} "
        f"peak_memory_reserved_bytes={peak_reserved_bytes}"
    )

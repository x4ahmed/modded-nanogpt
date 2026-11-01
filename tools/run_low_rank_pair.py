"""Run a fresh baseline/candidate pair in the same allocated GPU session.

The program loads before switching branches; it imports only the standard
library. It restores the candidate checkout even if either training run fails.
Results and exact commits are retained under logs/pairs/. No jobs are submitted.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "5be1ba0ca09293a007997cdce51c036bca3e89ae"
EXPERIMENT = "experiment/low-rank-parallel-mlp"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def run_env(seed):
    env = {key: value for key, value in os.environ.items() if not key.startswith("RTX_")}
    env.pop("NUM_SCHEDULED_ITERATIONS", None)
    env.update(PORTABLE="1", TRAIN_SEED=str(seed), RTX_MICROBATCH_TOKENS="16384", RTX_NGRAM_DIVISOR="32",
               RTX_VAL_SEQ_LEN="65536", RTX_EVAL_CE_SLAB_ROWS="4096", RTX_WARMUP_MICROBATCHES="2",
               RTX_MIN_HEADROOM_GIB="1.5", RTX_WANDB_UPLOAD="1")
    env.setdefault("WANDB_UPLOAD_PYTHON", str(Path.home() / "venvs/wandb-upload/bin/python"))
    return env


def read_result(log):
    text = log.read_text(errors="replace")
    matches = re.findall(r"final_val_loss=([\d.eE+-]+) training_time_ms=([\d.eE+-]+)", text)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one completed-training result in {log}")
    loss, time_ms = map(float, matches[0])
    if not math.isfinite(loss) or not math.isfinite(time_ms) or time_ms <= 0:
        raise RuntimeError(f"Invalid training result in {log}")
    # A completed short run must never masquerade as a full paired result.
    if not re.search(r"step:1194/1194 val_loss:", text):
        raise RuntimeError(f"Missing full 1194-update validation in {log}")
    return dict(final_val_loss=loss, training_seconds=time_ms / 1000)


def run_arm(branch, commit, seed, env):
    if git("status", "--porcelain"):
        raise RuntimeError("Checkout changed during the pair; refusing to switch branches.")
    git("switch", branch)
    if git("rev-parse", "HEAD") != commit:
        raise RuntimeError(f"{branch} changed after pair preparation")
    before = set((ROOT / "logs").glob("*.console.txt"))
    print(f"\nPAIR arm={branch} seed={seed} commit={commit}", flush=True)
    subprocess.run(["bash", "run_rtx.sh"], cwd=ROOT, env=env, check=True)
    new_logs = set((ROOT / "logs").glob("*.console.txt")) - before
    if len(new_logs) != 1:
        raise RuntimeError(f"Expected one console log, found {len(new_logs)}; do not train concurrently.")
    log = new_logs.pop()
    return dict(branch=branch, commit=commit, seed=seed, console_log=str(log.relative_to(ROOT)), **read_result(log))


def paired_deltas(baseline, candidate):
    return dict(loss_delta=candidate["final_val_loss"] - baseline["final_val_loss"],
                seconds_delta=candidate["training_seconds"] - baseline["training_seconds"],
                time_change_percent=100 * (candidate["training_seconds"] / baseline["training_seconds"] - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[999])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    if git("branch", "--show-current") != EXPERIMENT or git("status", "--porcelain"):
        raise RuntimeError(f"Start on a clean {EXPERIMENT} checkout.")
    if git("rev-parse", "baseline") != BASELINE_COMMIT:
        raise RuntimeError("Local baseline is not the frozen 5be1ba0 baseline; do not move it silently.")
    candidate_commit = git("rev-parse", "HEAD")
    if git("merge-base", "baseline", "HEAD") != BASELINE_COMMIT:
        raise RuntimeError("Candidate must descend from the frozen baseline.")
    # Compile/gradient checks and full-model smoke are gates, outside both timers.
    env = run_env(args.seeds[0])
    subprocess.run([sys.executable, "-u", "tests/check_low_rank_mlp.py"], cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, "-u", "tests/check_low_rank_optimizer.py"], cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, "-u", "tests/check_rtx5090.py", "--seed", str(args.seeds[0])],
                   cwd=ROOT, env=env, check=True)
    folder = ROOT / "logs" / "pairs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder.mkdir(parents=True, exist_ok=False)
    for seed in args.seeds:
        report = dict(experiment=EXPERIMENT, seed=seed, status="running", order=["baseline", "candidate"],
                      settings={k: v for k, v in run_env(seed).items() if k.startswith("RTX_") or k in ("TRAIN_SEED", "PORTABLE")})
        path = folder / f"seed-{seed}.json"
        try:
            report["baseline"] = run_arm("baseline", BASELINE_COMMIT, seed, run_env(seed))
            path.write_text(json.dumps(report, indent=2) + "\n")
            report["candidate"] = run_arm(EXPERIMENT, candidate_commit, seed, run_env(seed))
            report.update(paired_deltas(report["baseline"], report["candidate"]), status="complete")
        except BaseException as exc:
            report.update(status="failed", error=str(exc))
            raise
        finally:
            path.write_text(json.dumps(report, indent=2) + "\n")
            git("switch", EXPERIMENT)
            print(f"PAIR_RESULT {path}", flush=True)
        print(f"PAIR_COMPLETE seed={seed} loss_delta={report['loss_delta']:+.6f} "
              f"seconds_delta={report['seconds_delta']:+.3f} "
              f"time_change={report['time_change_percent']:+.3f}%", flush=True)


if __name__ == "__main__":
    main()

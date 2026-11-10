"""Phase 2 paired run: the frozen baseline, then this branch's candidate, in one GPU session.

Start on a clean checkout of an experiment branch that descends from the frozen baseline. For each
seed, the baseline commit trains first, then the candidate, with the same frozen settings; time and
loss are compared only within that pair. Gate first: the candidate's GPU smoke check, outside both
timers. The program loads before switching branches and imports only the standard library; it restores
the candidate checkout even if a run fails. Commits, results and console logs are kept in
logs/pairs/<UTC>/seed-<seed>.json. No jobs are submitted. Copy this file unchanged to each Phase 2 branch.

W&B: neither arm uploads itself (the frozen baseline would name itself plain "baseline"). After both
arms, both logs are imported under the pair's names, from the branch name without "experiment/":
baseline-for-<label>-1gpu-seed-<seed> and <label>-1gpu-seed-<seed>. Upload runs after training, never
on the clock; a failed upload keeps the local results and prints a retry command.

--candidate-env KEY=VALUE (repeatable, RTX_* only) sets a knob for the gates and the candidate arm only,
never the baseline: e.g. RTX_VOCAB_PRUNE_MASS=0.05 for one point of a sweep (give each point a --label).
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "baseline"
BASELINE_COMMIT = "5be1ba0ca09293a007997cdce51c036bca3e89ae"
FULL_UPDATES = 1194
# Gates on the candidate checkout before any training; "{seed}" becomes the first seed.
GATES = (("tests/check_rtx5090.py", "--seed", "{seed}"),)
UPLOADER = "tools/upload_wandb.py"
LABEL = re.compile(r"[a-z0-9][a-z0-9.-]*")


def pair_names(label, seed):
    """W&B display names: both arms carry the experiment label, so a baseline is never anonymous."""
    return dict(baseline=f"baseline-for-{label}-1gpu-seed-{seed}", candidate=f"{label}-1gpu-seed-{seed}")


def upload(arm, name, env):
    """Import one finished arm's console log under its pair name; True on success."""
    command = [env["WANDB_UPLOAD_PYTHON"], UPLOADER, arm["console_log"], "--entity", "i-learn",
               "--project", "speedrun", "--seed", str(arm["seed"]), "--training-commit", arm["commit"],
               "--name", name]
    try:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        print("WARNING: W&B upload failed; local results are safe. Retry: "
              + " ".join(shlex.quote(part) for part in command), flush=True)
        return False


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def run_env(seed):
    """Frozen local settings; inherited RTX_* and short-run flags are dropped, never passed through."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("RTX_")}
    env.pop("NUM_SCHEDULED_ITERATIONS", None)
    env.update(PORTABLE="1", TRAIN_SEED=str(seed), RTX_MICROBATCH_TOKENS="16384", RTX_NGRAM_DIVISOR="32",
               RTX_VAL_SEQ_LEN="65536", RTX_EVAL_CE_SLAB_ROWS="4096", RTX_WARMUP_MICROBATCHES="2",
               RTX_MIN_HEADROOM_GIB="1.5", RTX_WANDB_UPLOAD="0")  # the runner uploads under pair names
    env.setdefault("WANDB_UPLOAD_PYTHON", str(Path.home() / "venvs/wandb-upload/bin/python"))
    return env


def read_result(log):
    text = log.read_text(errors="replace")
    matches = re.findall(r"final_val_loss=([\d.eE+-]+) training_time_ms=([\d.eE+-]+)"
                         r"(?: peak_memory_allocated=([\d.]+) GiB)?", text)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one completed-training result in {log}")
    loss, time_ms = float(matches[0][0]), float(matches[0][1])
    if not math.isfinite(loss) or not math.isfinite(time_ms) or time_ms <= 0:
        raise RuntimeError(f"Invalid training result in {log}")
    # A completed short run must never masquerade as a full paired result.
    if not re.search(rf"step:{FULL_UPDATES}/{FULL_UPDATES} val_loss:", text):
        raise RuntimeError(f"Missing full {FULL_UPDATES}-update validation in {log}")
    result = dict(final_val_loss=loss, training_seconds=time_ms / 1000)
    if matches[0][2]:
        result["peak_allocated_gib"] = float(matches[0][2])
    return result


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
    parser.add_argument("--label", help="W&B experiment label; default: branch name without experiment/.")
    parser.add_argument("--candidate-env", action="append", default=[], metavar="KEY=VALUE",
                        help="RTX_* setting for the gates and the candidate arm only (repeatable).")
    args = parser.parse_args()
    candidate_env = dict(item.split("=", 1) for item in args.candidate_env if "=" in item)
    if len(candidate_env) != len(args.candidate_env) or not all(k.startswith("RTX_") for k in candidate_env):
        parser.error("--candidate-env takes RTX_KEY=VALUE")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    experiment = git("branch", "--show-current")
    if not experiment or experiment == BASELINE or git("status", "--porcelain"):
        raise RuntimeError("Start on a clean experiment-branch checkout (not baseline, not detached).")
    label = args.label or experiment.removeprefix("experiment/")
    if not LABEL.fullmatch(label) or label.startswith("baseline"):
        raise RuntimeError(f"Invalid W&B label {label!r}: lowercase letters, digits, '.', '-'; not baseline*.")
    if git("rev-parse", BASELINE) != BASELINE_COMMIT:
        raise RuntimeError("Local baseline is not the frozen 5be1ba0 baseline; do not move it silently.")
    candidate_commit = git("rev-parse", "HEAD")
    if candidate_commit == BASELINE_COMMIT or git("merge-base", BASELINE, "HEAD") != BASELINE_COMMIT:
        raise RuntimeError("Candidate must be a new commit descending from the frozen baseline.")
    env = run_env(args.seeds[0]) | candidate_env
    for gate in GATES:
        command = [part.format(seed=args.seeds[0]) for part in gate]
        subprocess.run([sys.executable, "-u", *command], cwd=ROOT, env=env, check=True)
    folder = ROOT / "logs" / "pairs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder.mkdir(parents=True, exist_ok=False)
    for seed in args.seeds:
        names = pair_names(label, seed)
        report = dict(experiment=experiment, label=label, seed=seed, status="running", order=["baseline", "candidate"],
                      candidate_env=candidate_env,
                      settings={k: v for k, v in run_env(seed).items() if k.startswith("RTX_") or k in ("TRAIN_SEED", "PORTABLE")})
        path = folder / f"seed-{seed}.json"
        try:
            report["baseline"] = run_arm(BASELINE, BASELINE_COMMIT, seed, run_env(seed))
            path.write_text(json.dumps(report, indent=2) + "\n")
            report["candidate"] = run_arm(experiment, candidate_commit, seed, run_env(seed) | candidate_env)
            report.update(paired_deltas(report["baseline"], report["candidate"]), status="complete")
        except BaseException as exc:
            report.update(status="failed", error=str(exc))
            raise
        finally:
            git("switch", experiment)
            # Every finished arm is imported under its pair name, also when the other arm failed.
            for arm in ("baseline", "candidate"):
                if arm in report:
                    report[arm].update(wandb_name=names[arm],
                                       wandb_uploaded=upload(report[arm], names[arm], run_env(seed)))
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"PAIR_RESULT {path}", flush=True)
        base, cand = report["baseline"], report["candidate"]
        print(f"PAIR_COMPLETE seed={seed} baseline_loss={base['final_val_loss']:.6f} "
              f"candidate_loss={cand['final_val_loss']:.6f} loss_delta={report['loss_delta']:+.6f} "
              f"baseline_s={base['training_seconds']:.3f} candidate_s={cand['training_seconds']:.3f} "
              f"seconds_delta={report['seconds_delta']:+.3f} time_change={report['time_change_percent']:+.3f}% "
              f"wandb={names['baseline']},{names['candidate']}", flush=True)


if __name__ == "__main__":
    main()

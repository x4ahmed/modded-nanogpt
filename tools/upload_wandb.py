"""Import one completed RTX-local-v2 log AFTER training. No GPU is needed."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


HEADER = re.compile(
    r"^rtx-local-v2: one GPU \((.+)\), (\d+)-token microbatches .*?"
    r"n-gram table ([\d,]+) rows \(1/(\d+) .*?validation (\d+)-token forwards, "
    r"MLP tile \d+x\d+, eval CE slab (\d+) rows, steps (\d+)/(\d+)$")
STEP = re.compile(r"^step:(\d+)/(\d+) (train_loss_per_token|val_loss):(\S+) "
                  r"train_time:(\S+)ms step_avg:(\S+)ms$")
FINAL = re.compile(r"^final_val_loss=(\S+) training_time_ms=(\S+) "
                   r"peak_memory_allocated=(\S+) GiB$")


def number(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid metric: {value}")
    return result


def parse_log(text):
    lines = text.splitlines()
    headers = [(i, HEADER.fullmatch(line)) for i, line in enumerate(lines)
               if HEADER.fullmatch(line)]
    if len(headers) != 1:
        raise ValueError("Expected exactly one RTX-local-v2 run header (not preflight).")
    start, header = headers[0]
    gpu, mb, rows, divisor, val_seq, slab, planned, scheduled = header.groups()
    planned, scheduled = int(planned), int(scheduled)
    config = dict(import_kind="post_training", gpu=gpu, microbatch_tokens=int(mb),
                  ngram_rows=int(rows.replace(",", "")), ngram_divisor=int(divisor),
                  val_seq_len=int(val_seq), eval_ce_slab_rows=int(slab),
                  planned_updates=planned, scheduled_updates=scheduled,
                  full_schedule=planned == scheduled)
    readout = [line for line in lines[start + 1:] if line.startswith("READOUT ")]
    config["readout_variant"] = "mudd"  # Historical logs predate the selector.
    if readout:
        match = re.fullmatch(r"READOUT variant=(mudd|attnres)", readout[0])
        if len(readout) != 1 or not match:
            raise ValueError("Invalid readout configuration.")
        config["readout_variant"] = match[1]
    history, finals = {}, []
    summary = {"canonical_mask_fallback": False}
    previous = -1
    for line in lines[start + 1:]:
        if line.startswith("rtx-local-v2:"):
            raise ValueError("Multiple run headers: provide a single run log.")
        match = STEP.fullmatch(line)
        if match:
            step, total, metric, loss, elapsed, average = match.groups()
            step = int(step)
            if int(total) != planned or not previous <= step <= planned:
                raise ValueError("Inconsistent or out-of-order optimizer steps.")
            previous = step
            row = history.setdefault(step, {"optimizer_step": step})
            key = "train/loss_per_token" if metric == "train_loss_per_token" else "validation/loss"
            if key in row:
                raise ValueError("Duplicate metric: possible concatenated runs.")
            row.update({key: number(loss), "timing/training_ms": number(elapsed),
                        "timing/step_average_ms": number(average)})
            if metric == "train_loss_per_token":
                # Validation at the same step may include extra finalization time.
                # Preserve the timestamp belonging to this training-loss observation.
                row["train/elapsed_ms"] = number(elapsed)
        match = FINAL.fullmatch(line)
        if match:
            finals.append(match.groups())
        if line.startswith("WARNING: background canonical mask build failed"):
            summary["canonical_mask_fallback"] = True
        match = re.fullmatch(r"peak memory allocated: (\d+) MiB reserved: (\d+) MiB", line)
        if match:
            summary.update(peak_allocated_mib=int(match[1]), peak_reserved_mib=int(match[2]))
        if line.startswith("MEMORY verdict:"):
            summary["memory_gate_passed"] = line.endswith("PASS")
    if len(finals) != 1 or "validation/loss" not in history.get(planned, {}):
        raise ValueError("Run is incomplete: require final validation and exactly one final summary.")
    loss, elapsed, allocated = map(number, finals[0])
    if abs(loss - history[planned]["validation/loss"]) > 1e-5:
        raise ValueError("Final summary and validation loss disagree.")
    summary.update(final_val_loss=loss, training_time_ms=elapsed,
                   peak_memory_allocated_gib=allocated, completed_updates=planned)
    return {"config": config, "summary": summary, "history": list(history.values())}


def upload(payload, args, sdk):
    # Do not label the uploader machine's stats or current checkout as training metadata.
    settings = sdk.Settings(disable_code=True, disable_git=True, x_disable_stats=True,
                            x_disable_meta=True, console="off")
    with sdk.init(project=args.project, entity=args.entity, name=args.name,
                  id="import-" + payload["config"]["source_sha256"][:24], resume="never",
                  job_type="post-training-import", mode="online", settings=settings,
                  config=payload["config"]) as run:
        run.define_metric("optimizer_step", hidden=True, summary="none")
        run.define_metric("training_seconds", hidden=True, summary="none")
        run.define_metric("training_minutes", hidden=True, summary="none")
        run.define_metric("train/loss_per_token", step_metric="optimizer_step")
        run.define_metric("train/loss_vs_seconds", step_metric="training_seconds")
        run.define_metric("train/loss_vs_minutes", step_metric="training_minutes")
        for row in payload["history"]:
            if "train/loss_per_token" in row:
                loss = row["train/loss_per_token"]
                seconds = row["train/elapsed_ms"] / 1000
                run.log({"optimizer_step": row["optimizer_step"],
                         "training_seconds": seconds, "training_minutes": seconds / 60,
                         "train/loss_per_token": loss,
                         "train/loss_vs_seconds": loss, "train/loss_vs_minutes": loss})
        run.summary.update(payload["summary"])
        url = run.url
    return url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="One completed UUID log or console log.")
    parser.add_argument("--project", default="speedrun")
    parser.add_argument("--entity", default="i-learn", help="W&B user/team owning the project.")
    parser.add_argument("--name", help="Display name, e.g. pilot-1gpu-seed-999.")
    parser.add_argument("--seed", type=int, help="Actual training seed; never inferred from current environment.")
    parser.add_argument("--training-commit", help="Actual training commit, not the uploader's checkout.")
    parser.add_argument("--dry-run", action="store_true", help="Preview JSON; no SDK, login or network required.")
    args = parser.parse_args()
    try:
        raw = args.log.read_bytes()
        payload = parse_log(raw.decode("utf-8", errors="replace"))
        payload["config"]["source_sha256"] = hashlib.sha256(raw).hexdigest()
        if args.seed is not None:
            payload["config"]["train_seed"] = args.seed
        if args.training_commit is not None:
            payload["config"]["training_commit_user_supplied"] = args.training_commit
        if args.dry_run:
            print(json.dumps(payload, indent=2, allow_nan=False))
            return 0
        try:
            import wandb
        except ImportError:
            parser.error("Install optional uploader dependency: python -m pip install wandb")
        print(f"W&B run: {upload(payload, args, wandb)}")
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Upload refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

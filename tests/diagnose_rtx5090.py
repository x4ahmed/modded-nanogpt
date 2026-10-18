"""Run all NaN diagnostics: python tests/diagnose_rtx5090.py

Each case gets a fresh process. Failures do not stop the remaining cases.
No training code or baseline settings are modified. Logs are diagnostic, not timings.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    "gemm", "mlp", "ce", "eval", "sampled", "full",
    "bf16_mlp_control", "bf16_logits_control", "trace",
)
MARKER = "RTX_DIAG_RESULT "


def execute(command, log_path, timeout):
    """Stream stdout/stderr, bound the run, and require a machine-readable result."""
    timed_out = threading.Event()
    result = None
    with log_path.open("x", encoding="utf-8") as logfile:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1,
                                start_new_session=os.name == "posix")

        def stop():
            if proc.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                except ProcessLookupError:
                    pass

        def expire():
            timed_out.set()
            stop()

        timer = threading.Timer(timeout, expire)
        timer.daemon = True
        timer.start()
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                logfile.write(line)
                logfile.flush()
                if line.startswith(MARKER):
                    try:
                        result = json.loads(line[len(MARKER):])
                    except json.JSONDecodeError:
                        result = None
            code = proc.wait()
        finally:
            timer.cancel()
            stop()
            proc.wait()
            proc.stdout.close()
    passed = (not timed_out.is_set() and code == 0 and isinstance(result, dict)
              and result.get("status") == "pass")
    return {"status": "pass" if passed else "fail", "returncode": code,
            "timed_out": timed_out.is_set(), "result": result, "log": str(log_path)}


def worker(args):
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    # Keep stale debug, distributed, or schedule overrides out of this experiment.
    for key in list(os.environ):
        if key.startswith("RTX_"):
            del os.environ[key]
    os.environ.pop("NUM_SCHEDULED_ITERATIONS", None)
    os.environ.update(PORTABLE="1", TRAIN_SEED=str(args.seed), RTX_MICROBATCH_TOKENS=str(args.tokens),
                      RTX_NGRAM_DIVISOR="32", RTX_VAL_SEQ_LEN="65536", RTX_EVAL_CE_SLAB_ROWS="4096",
                      RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1",
                      MASTER_PORT="29517")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    result = {"case": args.case, "status": "fail"}
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Allocate exactly one visible CUDA GPU.")
        name = torch.cuda.get_device_name(0)
        if "5090" not in name or torch.cuda.get_device_capability(0) != (12, 0):
            raise RuntimeError(f"Expected RTX 5090 / SM120, found {name}")
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        print(f"CASE {args.case} GPU={name} torch={torch.__version__} CUDA={torch.version.cuda}", flush=True)
        print(f"seed={args.seed} tokens={args.tokens} ngram_divisor=32 val_seq_len=65536", flush=True)
        from rtx_diagnostic_cases import run_case
        run_case(args.case, args.tokens)
        torch.cuda.synchronize()
        result["status"] = "pass"
    except Exception as exc:
        traceback.print_exc()
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_initialized():
                    print(f"MEMORY peak_allocated_bytes={torch.cuda.max_memory_allocated()} "
                          f"peak_reserved_bytes={torch.cuda.max_memory_reserved()}", flush=True)
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
            except Exception as exc:
                print(f"Cleanup warning: {exc}", flush=True)
        print(MARKER + json.dumps(result), flush=True)
    return 0 if result["status"] == "pass" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, help="Run only this case (normally leave unset).")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--tokens", type=int, choices=(2048, 16384), default=16384)
    parser.add_argument("--timeout", type=int, default=1800, help="Maximum seconds per case.")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.case:
        return worker(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = ROOT / "logs" / f"rtx-diagnostics-{stamp}-{os.getpid()}"
    folder.mkdir(parents=True, exist_ok=False)
    print(f"Diagnostic output: {folder}\nNine isolated cases; compilation may take tens of minutes.", flush=True)
    metadata = {"seed": args.seed, "tokens": args.tokens, "cases": {}}
    for label, command in (("commit", ["git", "rev-parse", "HEAD"]), ("working_tree", ["git", "status", "--short"])):
        info = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        metadata[label] = info.stdout.strip() if info.returncode == 0 else info.stderr.strip()
    for case in CASES:
        print(f"\n===== {case} =====", flush=True)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--case", case,
                   "--seed", str(args.seed), "--tokens", str(args.tokens)]
        metadata["cases"][case] = execute(command, folder / f"{case}.txt", args.timeout)
        (folder / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\n===== SUMMARY =====", flush=True)
    for case, result in metadata["cases"].items():
        print(f"{case}: {result['status']}" + (" (timeout)" if result["timed_out"] else ""))
    passed = all(r["status"] == "pass" for r in metadata["cases"].values())
    print(f"RTX_DIAGNOSTICS status={'pass' if passed else 'fail'} summary={folder / 'summary.json'}")
    print("Controls change arithmetic only in their test process. Passing controls do not validate normal training.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

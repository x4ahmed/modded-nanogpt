"""Run inside a one-RTX-5090 allocation: python tests/check_rtx5090.py

Exercises the existing trainer: all-stage warmup, reset, two complete updates,
then full validation. Checks are installed only in this test process. This is
a numerical/memory smoke test, not a convergence or upstream-parity proof.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


class Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, text):
        self.console.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.logfile.flush()


def configure(args):
    # Stale debug flags must not turn this into a different execution path.
    for key in list(os.environ):
        if key.startswith("RTX_"):
            del os.environ[key]
    os.environ.pop("NUM_SCHEDULED_ITERATIONS", None)
    os.environ.update(PORTABLE="1", TRAIN_SEED=str(args.seed), RTX_MAX_STEPS="2",
                      RTX_MICROBATCH_TOKENS="16384", RTX_NGRAM_DIVISOR="32",
                      RTX_VAL_SEQ_LEN=str(args.val_seq_len), RTX_EVAL_CE_SLAB_ROWS="4096",
                      RTX_WARMUP_MICROBATCHES="2", RTX_MIN_HEADROOM_GIB="1.5")
    os.environ.update(RTX_ROUTING=args.routing, RTX_ATTNRES_BLOCK_SIZE=str(args.block_size),
                      RTX_ATTNRES_INJECT=str(args.attnres_inject))


def run_check(args):
    configure(args)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Allocate exactly one visible CUDA GPU first.")
    name = torch.cuda.get_device_name(0)
    if "5090" not in name or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError(f"Expected RTX 5090 / SM120; found {name}.")
    print(f"DEVICE {name}; torch={torch.__version__}; CUDA={torch.version.cuda}")
    for command in (["git", "rev-parse", "HEAD"], ["git", "status", "--short"]):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        print(f"{' '.join(command)}: {result.stdout.strip() or result.stderr.strip() or 'clean'}")
    print("SETTINGS " + " ".join(f"{k}={v}" for k, v in sorted(os.environ.items())
                                  if k.startswith("RTX_") or k in ("TRAIN_SEED", "DATA_PATH", "PORTABLE")))
    print("Diagnostic run only. Timings include numerical checks. Compilation may take several minutes.")

    # The trainer snapshots its entry script and package relative to that script.
    # Present its own path while importing, not this nested test's path.
    entry_path = sys.argv[0]
    try:
        sys.argv[0] = str(ROOT / "train_gpt_rtx.py")
        import train_gpt_rtx as trainer
    finally:
        sys.argv[0] = entry_path

    state = {"reset": False, "updates": 0, "final_loss": None, "memory": False, "routing_updated": False}

    def finite(label, tensor):
        if tensor is None or not tensor.is_floating_point() or tensor.numel() == 0:
            return
        # Reductions avoid allocating a table-sized boolean mask or fp32 copy.
        tensor = tensor.detach()
        low, high = torch.aminmax(tensor)
        if not (math.isfinite(low.item()) and math.isfinite(high.item())):
            raise RuntimeError(f"Non-finite {label}")

    original_log = trainer.start_run_log

    def checked_log(*args, **kwargs):
        print0, flush = original_log(*args, **kwargs)

        def checked_print(message, console=False):
            print0(message, console=console)
            if message == "Resetting model":
                state["reset"] = True
            if message.startswith(("step:", "final_val_loss=")):
                match = re.search(r"(?:val_loss:|final_val_loss=)(\S+)", message)
                if match:
                    loss = float(match.group(1))
                    if not math.isfinite(loss):
                        raise RuntimeError(f"Non-finite validation loss: {loss}")
                    if message.startswith("final_val_loss="):
                        state["final_loss"] = loss
        return checked_print, flush

    trainer.start_run_log = checked_log
    original_collect = trainer.GradAccumulator.collect

    def checked_collect(accumulator):
        if args.routing == "block_attnres":
            required = ("attnres_queries", "attnres_ve_gate") + (("attnres_last_inject",) if args.attnres_inject else ())
            for param in accumulator.params:
                if getattr(param, "label", "") in required and param.grad is None:
                    raise RuntimeError(f"Missing routing gradient: {param.label}")
        for i, param in enumerate(accumulator.params):
            finite(f"parameter gradient {i}", param.grad)
        original_collect(accumulator)

    trainer.GradAccumulator.collect = checked_collect
    original_release = trainer.GradAccumulator.release

    def checked_release(accumulator, do_adam):
        original_release(accumulator, do_adam)
        for i, param in enumerate(accumulator.params):
            finite(f"accumulated gradient {i}", param.grad)

    trainer.GradAccumulator.release = checked_release
    original_accumulate = trainer.LocalNgramTable.accumulate_grad

    def checked_accumulate(table, row_ids, grad):
        if grad is None:
            raise RuntimeError("Missing n-gram sink gradient")
        finite("n-gram sink gradient", grad)
        return original_accumulate(table, row_ids, grad)

    trainer.LocalNgramTable.accumulate_grad = checked_accumulate
    original_step = trainer.Trainer.train_step

    def checked_step(instance, step, batches, tables):
        routing_before = None
        if args.routing == "block_attnres" and state["reset"] and step % 2 == 1:
            routing_before = instance.uncompiled.attnres_queries.detach().clone()
        compiled_model = instance.model
        microbatch = 0

        def checked_forward(*args, **kwargs):
            nonlocal microbatch
            loss = compiled_model(*args, **kwargs)
            finite(f"forward loss at step {step}, microbatch {microbatch}", loss)
            microbatch += 1
            return loss

        original_update = instance.training_manager.step_optimizers

        def checked_update(*args, **kwargs):
            finite("n-gram accumulated gradient", instance.ngram.grad)
            finite("value-embedding accumulated gradient", instance.value_embeds.grad_accum)
            return original_update(*args, **kwargs)

        instance.model = checked_forward
        instance.training_manager.step_optimizers = checked_update
        try:
            loss = original_step(instance, step, batches, tables)
        finally:
            instance.model = compiled_model
            instance.training_manager.step_optimizers = original_update
        for name, param in instance.uncompiled.named_parameters():
            finite(f"weight {name} after step {step}", param)
        finite("n-gram table after update", instance.ngram.table)
        finite("n-gram second moment", instance.ngram.exp_avg_sq)
        if state["reset"]:
            state["updates"] += 1
        if routing_before is not None:
            if torch.equal(routing_before, instance.uncompiled.attnres_queries.detach()):
                raise RuntimeError("AttnRes queries did not change on the Adam update")
            state["routing_updated"] = True
        print(f"CHECK {'TRAIN' if state['reset'] else 'WARMUP'} step={step} microbatches={microbatch} finite=yes")
        return loss

    trainer.Trainer.train_step = checked_step
    original_verdict = trainer.MemoryReport.verdict

    def checked_verdict(report):
        ok = original_verdict(report)
        if not ok:
            raise RuntimeError("GPU memory headroom is below 1.5 GiB")
        state["memory"] = True
        return ok

    trainer.MemoryReport.verdict = checked_verdict
    trainer.main()
    if state["updates"] != 2 or state["final_loss"] is None or not state["memory"]:
        raise RuntimeError(f"Incomplete verification: {state}")
    if args.routing == "block_attnres" and not state["routing_updated"]:
        raise RuntimeError("Routing optimizer update was not verified")
    print(f"RTX5090_CHECK status=pass updates=2 final_val_loss={state['final_loss']:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--routing", choices=("baseline", "static", "block_attnres"), default="baseline")
    parser.add_argument("--block-size", type=int, choices=range(1, 18), default=6)
    parser.add_argument("--attnres-inject", type=int, choices=(0, 1), default=1,
                        help="Block mode: keep the x0/n-gram injections and layer-6 skip (1) or drop them (0).")
    parser.add_argument("--val-seq-len", type=int, choices=(65536, 262144), default=65536,
                        help="Diagnostic validation packing; default 65536, not a frozen baseline.")
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    (ROOT / "logs").mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / "logs" / f"check-rtx5090-{stamp}-{os.getpid()}.txt"
    print(f"Verification log: {path}")
    with path.open("x", encoding="utf-8") as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            try:
                run_check(args)
            except (Exception, KeyboardInterrupt):
                traceback.print_exc()
                print("RTX5090_CHECK status=fail")
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

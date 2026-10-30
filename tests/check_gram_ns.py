"""GPU-only Gram-NS diagnostics, no dataset or training run required.

python -u tests/check_gram_ns.py
Tests the new kernel, full-size cascades, and complete captured bank updates.
Batch 24 models one RTX rank; batch 3 models one of eight H100 ranks, WITHOUT
collectives. Timings are diagnostics, not end-to-end speed or convergence proof.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check_rtx5090 import Tee


def run_checks(args):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Run with exactly one visible CUDA GPU; no dataset is needed.")
    print(f"DEVICE name={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()} "
          f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    print("Compiling optimizer kernels. Each completed case prints a result.", flush=True)
    from track_1_short.optim.anvil import ANVIL_MAPS, AnvilBank, anvil_cascade
    from track_1_short.optim.gram_newton_schulz import gram_pair_cascade
    from track_1_short.perf.cuda_graphs.optimizer_graphs import AnvilBankGraphs
    from track_1_short.perf.kernels.gram_product import sym_product
    from track_1_short.perf.kernels.polar_express import XTX

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False  # independent FP32 kernel reference only
    device = torch.device("cuda", 0)

    def finite(label, x):
        if not bool(x.isfinite().all()):
            raise AssertionError(f"Non-finite {label}")

    def compare(label, x, reference, max_rel=None):
        finite(label, x)
        finite(label + " reference", reference)
        xf, rf = x.float().flatten(), reference.float().flatten()
        rel = ((xf-rf).norm() / rf.norm().clamp_min(1e-20)).item()
        cosine = torch.nn.functional.cosine_similarity(xf, rf, dim=0).item()
        print(f"NUMERICS case={label} rel_l2={rel:.6g} cosine={cosine:.6g}", flush=True)
        if not math.isfinite(rel) or (max_rel is not None and rel > max_rel):
            raise AssertionError(f"{label}: relative error {rel} exceeds {max_rel}")

    # Odd sizes, batched inputs, noncontiguous operands, diagonal epilogue and
    # beta=0 with a NaN C: ignored addends must not leak into the result.
    for shape in ((129, 129), (3, 256, 256), (3, 768, 768)):
        for beta in (0., -.25):
            a = torch.randn(shape, device=device).bfloat16().mT
            b = torch.randn(shape, device=device).bfloat16()
            c = torch.randn(shape, device=device).bfloat16()
            if beta == 0.:
                c.fill_(float("nan"))
            out = torch.empty_like(b)
            sym_product(a, b, c, alpha=.5, beta=beta, diagonal=1.25, out=out)
            reference = .5 * (a.float() @ b.float())
            if beta:
                reference.add_(c.float(), alpha=beta)
            reference.diagonal(dim1=-2, dim2=-1).add_(1.25)
            reference = (reference.tril() + reference.tril(-1).mT).bfloat16()
            compare(f"kernel-{shape}-beta{beta}", out, reference, .004)
            assert torch.equal(out, out.mT), "triangular stores must produce exact symmetry"
    print("KERNEL_PARITY status=pass", flush=True)

    def scalar(v):
        return torch.tensor(v, dtype=torch.float32, device=device)

    # Each precision stress case uses the actual production MLP matrix shape.
    # The baseline comparison reports numerical drift, not a quality guarantee.
    for kind in ("gaussian", "low_rank", "ill_conditioned", "tiny", "zero"):
        torch.manual_seed(args.seed)
        if kind == "low_rank":
            g = torch.randn(1, 2816, 8, device=device) @ torch.randn(1, 8, 768, device=device)
        else:
            g = torch.randn(1, 2816, 768, device=device)
        if kind == "ill_conditioned":
            g *= torch.logspace(0, -6, 768, device=device)
        elif kind == "tiny":
            g *= 1e-8
        elif kind == "zero":
            g.zero_()
        g = g.bfloat16()
        before = g.clone()
        velocities = [torch.zeros(2, *g.shape, device=device) for _ in range(2)]
        results = [anvil_cascade(g, v, scalar(.85), True, scalar(.85), scalar(.4385), gram_pairs=paired)
                   for v, paired in zip(velocities, (False, True))]
        compare(kind, results[1], results[0], .06 if kind == "gaussian" else None)
        assert torch.equal(g, before), "whitening mutated its gradient input"
        assert torch.equal(*velocities), "momentum rails changed with whitening mode"
        if kind == "zero":
            assert results[1].count_nonzero().item() == 0
        # Estimate the largest singular value without a full costly SVD.
        matrix = results[1][0].float()
        direction = torch.randn(768, device=device)
        for _ in range(20):
            direction = matrix.T @ (matrix @ direction)
            direction /= direction.norm().clamp_min(1e-20)
        spectral = (matrix @ direction).norm().item()
        print(f"SPECTRAL case={kind} estimated_max={spectral:.6g}", flush=True)
        assert math.isfinite(spectral) and spectral < 2., "whitening amplification is unstable"

    # Independently rounded dense FP32 recurrence versus the fused GPU path,
    # and compiled versus eager helper. No optimizer momentum in this check.
    def dense_product(a, b, c, *, alpha, beta, diagonal, out):
        y = alpha * (a.float() @ b.float())
        if beta:
            y.add_(c.float(), alpha=beta)
        y.diagonal(dim1=-2, dim2=-1).add_(diagonal)
        return out.copy_(y.tril() + y.tril(-1).mT)

    def dense_gram(x, *, out):
        return out.copy_(x.float().mT @ x.float())

    x = torch.randn(1, 2816, 768, device=device)
    x = (x / (1.05*x.norm()+1e-6)).bfloat16()
    gram = torch.empty(1, 768, 768, device=device, dtype=torch.bfloat16)
    XTX(x, out=gram)
    expected = gram_pair_cascade(x.clone(), gram.clone(), ANVIL_MAPS,
                                gram_out=dense_gram, sym_product_out=dense_product)
    actual = gram_pair_cascade(x.clone(), gram.clone(), ANVIL_MAPS,
                              gram_out=XTX, sym_product_out=sym_product)
    compare("paired-dense-reference", actual, expected, .02)

    def paired_core(x, gram):
        return gram_pair_cascade(x, gram, ANVIL_MAPS, gram_out=XTX, sym_product_out=sym_product)
    compiled = torch.compile(paired_core, fullgraph=True, dynamic=False)
    compare("paired-compiled-eager", compiled(x.clone(), gram.clone()), actual, .001)
    print("CASCADE_NUMERICS status=pass", flush=True)

    def make_bank(batch, paired):
        torch.manual_seed(args.seed)
        shape = (batch, 2816, 768)
        grad = torch.randn(shape, device=device).bfloat16()
        grad[-1].zero_()  # representative frozen slot
        lr = torch.full((batch, 1, 1), .01, device=device)
        lr[-1].zero_()
        return AnvilBank(label="mlp_bank", grad=grad,
                         velocity=torch.zeros(2, *shape, device=device),
                         lane_energy=torch.zeros(batch, 2816, 1, device=device),
                         mantissa=torch.zeros(shape, device=device, dtype=torch.uint16),
                         p_slice=(torch.randn(shape, device=device)*.02).bfloat16(),
                         momentum=scalar(.85), eff_wd=scalar(.01), fast_beta=scalar(.85),
                         fast_weight=scalar(1.), eff_lr=lr, beta2=.95,
                         split_baddbmm=True, red_dim=-1, gram_pairs=paired)

    for batch in args.batches:
        print(f"BANK_CHECK matrices={batch} shape=2816x768 no_collectives=true", flush=True)
        banks = [make_bank(batch, flag) for flag in (False, True)]
        runners = [AnvilBankGraphs(device), AnvilBankGraphs(device)]
        for bank, runner in zip(banks, runners):
            original_frozen = bank.p_slice[-1].clone()
            runner.capture([bank])
            runner.update(bank)  # production self-check compares every mutated state bitwise
            runner.seal()
            # Check reset by copying state back in place, then use different live
            # scalar values to detect graphs that bake the first stage's constants.
            bank.velocity.zero_()
            bank.lane_energy.zero_()
            bank.mantissa.zero_()
            bank.fast_beta.fill_(.78)
            bank.fast_weight.fill_(.4385)
            bank.momentum.fill_(.82)
            bank.eff_lr[:-1].fill_(.005)
            runner.unchecked.add(bank.label)
            runner.update(bank)
            assert torch.equal(bank.p_slice[-1], original_frozen), "frozen weight changed"
            for key, tensor in bank.mutated().items():
                if tensor.is_floating_point():
                    finite(key, tensor)
            for _ in range(5):
                runner.update(bank)
        torch.cuda.synchronize()
        samples = [[], []]
        for repeat in range(args.repeats):
            # Alternate order to reduce drift from thermals or clock changes.
            for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                runners[index].update(banks[index])
                end.record()
                end.synchronize()
                samples[index].append(start.elapsed_time(end))
        baseline_ms, paired_ms = map(statistics.median, samples)
        print(f"BANK_TIMING matrices={batch} standard_ms={baseline_ms:.6f} gram_pair_ms={paired_ms:.6f} "
              f"ratio={paired_ms/baseline_ms:.4f} samples={args.repeats} "
              f"faster={'yes' if paired_ms < baseline_ms else 'no'}", flush=True)
        print(f"GRAPH_REPLAY matrices={batch} status=pass reset_and_live_scalars=true", flush=True)
        del runners, banks, runner, bank
        torch.cuda.empty_cache()
    print("Kernel/bank timing only. Run the RTX smoke test before a same-seed full training comparison.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--batches", type=int, nargs="+", default=[3, 24], choices=(3, 24))
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args(argv)
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    os.chdir(ROOT)
    (ROOT / "logs").mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / "logs" / f"check-gram-ns-{stamp}-{os.getpid()}.txt"
    print(f"Verification log: {path}", flush=True)
    with path.open("x", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                run_checks(args)
            except (Exception, KeyboardInterrupt):
                traceback.print_exc()
                # OOM is caught here too; a crash must never look like a passing gate.
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_initialized():
                    try:
                        free, total = torch.cuda.mem_get_info()
                        print(f"MEMORY allocated_bytes={torch.cuda.memory_allocated()} "
                              f"reserved_bytes={torch.cuda.memory_reserved()} free_bytes={free} total_bytes={total}")
                    except Exception:
                        print("MEMORY unavailable after CUDA error")
                print("GRAM_NS_CHECK status=fail", flush=True)
                return 1
            print("GRAM_NS_CHECK status=pass", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

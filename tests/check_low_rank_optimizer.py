"""Check rank-384 optimizer kernels and complete updates before model warmup.

Run on the allocated GPU: python -u tests/check_low_rank_optimizer.py
No dataset or distributed process group is needed. No training timing is measured.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    import triton
    from low_rank_reference import ParityReport
    from track_1_short.optim.anvil import AnvilBank, anvil_bank_update, ANVIL_MAPS
    from track_1_short.perf.kernels.polar_express import XTX, ba_plus_cAA, xtx_num_stages

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    torch.manual_seed(999)
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties("cuda")
    limit = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
    print(f"LOW_RANK_OPTIMIZER_CHECK device={props.name} torch={torch.__version__} "
          f"triton={triton.__version__} shared_limit={limit} xtx_stages={xtx_num_stages(384, limit)}", flush=True)
    report = ParityReport()

    with torch.no_grad():
        for label, rows in (("a", 2816), ("b", 768)):
            shape = (1, rows, 384)
            grad = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * .01
            gram = torch.empty((1, 384, 384), device="cuda", dtype=torch.bfloat16)
            print(f"OPTIMIZER_PROBE factor={label} shape={shape} stage=xtx", flush=True)
            XTX(grad, out=gram)
            reference = (grad.float().mT @ grad.float()).bfloat16()
            report.compare(f"optimizer/{label}/xtx", gram, reference, .01)
            report.compare(f"optimizer/{label}/symmetry", gram, gram.mT, 0.)
            poly = torch.empty_like(gram)
            _, b, c = ANVIL_MAPS[0]
            print(f"OPTIMIZER_PROBE factor={label} stage=polynomial", flush=True)
            ba_plus_cAA(gram, alpha=c, beta=b, out=poly)
            reference = (c * (gram.float() @ gram.float().mT) + b * gram.float()).bfloat16()
            report.compare(f"optimizer/{label}/polynomial", poly, reference, .01)

            scalar = lambda value: torch.tensor(value, device="cuda", dtype=torch.float32)
            p = torch.zeros_like(grad) if label == "a" else torch.randn_like(grad) * .03
            bank = AnvilBank(
                label=f"low_rank_{label}", grad=torch.zeros_like(grad),
                velocity=torch.zeros((2, *shape), device="cuda", dtype=torch.float32),
                lane_energy=torch.zeros((1, 1, 384), device="cuda", dtype=torch.float32),
                mantissa=torch.zeros(shape, device="cuda", dtype=torch.uint16), p_slice=p,
                momentum=scalar(.85), eff_wd=scalar(0.), fast_beta=scalar(.85), fast_weight=scalar(1.),
                eff_lr=torch.full((1, 1, 1), .001, device="cuda"), beta2=.95,
                split_baddbmm=rows > 1024, red_dim=-2)
            initial = p.clone()
            print(f"OPTIMIZER_PROBE factor={label} stage=compiled-zero-update", flush=True)
            anvil_bank_update(bank)
            report.compare(f"optimizer/{label}/zero-update", p, initial, 0.)
            for index, fast_weight in enumerate((1., .4385)):
                print(f"OPTIMIZER_PROBE factor={label} stage=compiled-nonzero-update rail_weight={fast_weight}", flush=True)
                bank.grad.copy_(grad)
                bank.fast_weight.fill_(fast_weight)
                anvil_bank_update(bank)
                torch.cuda.synchronize()
                for name, value in bank.mutated().items():
                    if value.is_floating_point() and not torch.isfinite(value).all():
                        raise AssertionError(f"Nonfinite {label}/{index}/{name}")
                if not (bank.velocity.count_nonzero().item() and bank.lane_energy.count_nonzero().item()):
                    raise AssertionError(f"Optimizer state did not advance for {label}")
            if torch.equal(p, initial):
                raise AssertionError(f"Parameter {label} did not update")
    report.finish()
    print("LOW_RANK_OPTIMIZER_CHECK status=pass factors=a,b", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("LOW_RANK_OPTIMIZER_CHECK status=fail", flush=True)
        raise

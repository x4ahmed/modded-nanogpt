# RTX-local-v1 CE decision

Status: **RESOLVED — target `90` kept.** Verified on an RTX 5090 (`tujestpolin`,
2026-08-30, one rank, torch 2.10.0+cu128).

```bash
torchrun --standalone --nproc_per_node=1 tests/rtx_preflight.py --seed 999
```

The fused CE kernel's compute target stays at `90`. PyTorch emits PTX, and the
sm_90 PTX JITs onto SM120 without change. No fallback was needed or added.

## Measured evidence

| Stage | Result |
|---|---|
| `CE_COMPILE` | pass, `target=90` |
| `CE_SHARED_CONFIG` | pass — `dynamic=100608`, `static_min=64`, `total_min=100672`, device opt-in limit `101376` |
| `CE_LAUNCH` | pass |
| `CE_SYNC` | pass |
| `CE_PARITY` | pass — `loss_max_abs=2.86e-06`, `grad_rel_l2=0`, `grad_cosine=1`, `grad_exact_fraction=1` |

The kernel fits with **704 bytes** of headroom against SM120's per-block opt-in
ceiling, and the FP8 logit gradients matched the eager reference exactly. The
margin is real but thin: any increase in `CE_KERNEL_VOCAB_SIZE` or in the
kernel's static shared memory would exceed the limit on this hardware.

## Related finding from the same run

The CE kernel fits; Inductor's **FlexAttention** template did not. Its default
tiles at `QK_HEAD_DIM=128` requested `180,248 B` against the same `101,376 B`
ceiling, failing with "No valid triton configs". Fixed by pinning tile sizes in
`portable_attention.KERNEL_OPTIONS` — see that module for the reasoning. Same
hardware limit, different kernel, and unrelated to the CE compute target.

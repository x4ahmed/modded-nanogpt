# RTX-local-v1 CE decision

Status: **PENDING RTX 5090 preflight**

```bash
torchrun --standalone --nproc_per_node=2 tests/rtx_preflight.py --seed 7
```

Keep the fused CE kernel's compute target at `90` until the SM120 preflight provides
device evidence. The kernel requests `100,608 B` dynamic shared memory and declares at
least `64 B` static shared memory (`100,672 B` minimum total). No fallback is authorized.

The executable preflight records these stages separately:

| Stage | Required evidence |
|---|---|
| `CE_COMPILE` | target `90` compiles/loads on SM120 |
| `CE_SHARED_CONFIG` | `set_shared_memory_config(100608)` succeeds; opt-in limit logged |
| `CE_LAUNCH` | minimal raw kernel launch enqueues |
| `CE_SYNC` | device synchronization succeeds |
| `CE_PARITY` | loss and FP8 logit-gradient metrics pass the eager reference |

Decision after the run: keep target `90` only if every stage passes. Otherwise record the
first failing stage and investigate that cause before changing the target or kernel.

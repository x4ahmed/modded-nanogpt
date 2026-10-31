# Low-rank parallel MLP

Branch: `experiment/low-rank-parallel-mlp`, directly from frozen `baseline` (`5be1ba0`).
Only the extra MLP at transformer position 8 (bank slot 11) changes. Indices are zero-based.

| Setting | Baseline | Experiment |
| --- | --- | --- |
| Up projection + activation | 768 -> 2816, squared ReLU | Unchanged |
| Selected down projection | 2816 -> 768 | 2816 -> 384 -> 768, no activation between factors |
| Other ten MLPs / attention / routing | Baseline | Unchanged |
| Training / validation arithmetic | FP8 / BF16 | FP8 / BF16; new bottleneck uses a current amax scale |
| Initialization | Zero down weight | A = zero; B = random nonzero; output remains zero |
| Optimizer | ANVIL | Both factors use ANVIL every step, base LR/WD unchanged, down-projection 2x LR multiplier plus existing shape scaling |
| Schedule / validation | 1194 updates / fixed packing | Unchanged |

Both factors follow the MLP tail-average and bank-blend windows. Each factor gets its own
norm restoration. Their factorwise optimization and averaging are part of this experiment;
they are not mathematically equivalent to optimizing/averaging a dense product.

New factor initialization preserves the RNG state. All common parameters keep their
baseline initialization draws. Warmup resets restore both factors and refresh FP8 caches.
The old dense down matrix stays as frozen zero padding to preserve the original bank layout.
Its buffers and bank work remain allocated: this experiment does not promise lower memory.

On eight GPUs, the two small factor matrices use averaged gradient all-reduces and replicated
ANVIL updates. This avoids padding each single factor into an eight-matrix bank. The shared
model, optimizer capture and FP8-refresh capture support this path, but actual eight-GPU
training is untested. One-GPU CPU tests do not establish an H100 speedup.

## Expected saving and limits

The selected down projection uses 36.4% fewer multiply-accumulates. Including its unchanged
up projection, that MLP uses 18.2% less matrix arithmetic: about 1.65% across all eleven MLPs.
Extra quantization, smaller GEMMs and factor optimizer work can erase the saving. Measure
the complete run; no speed or quality improvement is established yet.

## Run one fresh baseline/candidate pair

Use one allocated RTX 5090, with no other training process on that GPU. From the cluster checkout:

```bash
source ~/venvs/nanogpt/bin/activate
cd ~/speed-run/modded-nanogpt
git fetch origin
git switch experiment/low-rank-parallel-mlp
git pull --ff-only origin experiment/low-rank-parallel-mlp
git show-ref --verify --quiet refs/heads/baseline || git branch baseline origin/baseline
python -u tools/run_low_rank_pair.py --seeds 999
```

The runner refuses a dirty checkout or a moved baseline. It clears stale RTX/short-run flags,
sets the frozen local settings and runs these steps, stopping on failure:

1. Focused low-rank FP8 eager/compiled forward/backward check.
2. Full-model smoke: all-stage warmup, reset, two updates, full validation and memory gate.
3. Full baseline seed 999, then full candidate seed 999, in the same allocation.

It returns to the experiment branch. W&B uploads happen after each successful training run:
`baseline-1gpu-seed-999` and `low-rank-parallel-r384-1gpu-seed-999`.
Console logs stay in `logs/*.console.txt`. Pair metadata and loss/time deltas are in
`logs/pairs/<UTC timestamp>/seed-999.json`; the original trainer logs remain in `logs/<UUID>.txt`.
Repeated names are separate W&B run IDs. The JSON associates each pair with its precise console logs.

After a promising screening pair, use `--seeds 999 1000 1001 1002 1003` for five fresh pairs.
Compare the paired differences; historical timings are context only. Do not stack experiments.

## Verification

CPU: `python -m unittest discover -s tests -p 'test_*.py' -v`.
The CPU custom-autograd test substitutes exact arithmetic for GPU kernels; it checks algebra,
rectangular shapes and folded-lambda gradients, not FP8 numerical accuracy.
GPU checks: `python -u tests/check_low_rank_mlp.py` then `python -u tests/check_rtx5090.py`.
GPU precision, full-model memory, convergence and end-to-end timing remain cluster checks.

The focused check uses production bootstrap scales and identical eager/compiled inputs.
An independent quantized PyTorch oracle checks all outputs and gradients at 3% relative L2.
The BF16 comparison retains 10% output / 15% tensor-gradient limits. For the scalar
post-lambda, a random signed dot product can nearly cancel, so its raw drift is printed
as a diagnostic; a reference-aligned, non-cancelling probe gates BF16 parity at the original
15% limit. Random scalar parity is still gated against the quantized oracle. These are
numerical screening tolerances, not evidence of convergence or quality equivalence.

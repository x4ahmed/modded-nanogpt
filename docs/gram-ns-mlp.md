# Paired Gram Newton–Schulz for the MLP optimizer

Branch: `experiment/gram-ns-mlp`, from frozen `baseline` (`5be1ba0`).
Status: implemented; RTX 5090 training and H100 distributed validation pending.

Local verification: 36 CPU/Bash tests passed. The fused kernel compiled offline
for SM90/SM120 and passed six runtime cases on RTX 3050 (Triton 3.4, torch 2.8).
The full optimizer test needs the cluster: the local Windows C compiler/toolkit
is missing. Those kernel checks do not validate the complete training pipeline.

Only the MLP whitening computation changes. Six ANVIL polynomial maps run in
three pairs. Each pair works on 768×768 Gram matrices and applies one combined
transform to the 2816×768 update. The Gram is rebuilt between pairs.

BF16, all coefficients, initial Gram/trace normalization, momentum rails, lane
equalization, decay, mantissa tracking and matrix grouping remain as in baseline.
Q/K, V/O, Adam, model architecture, loss, data, schedule and validation are unchanged.
No earlier experiment is included.

This follows [Gram Newton–Schulz](https://github.com/Dao-AILab/gram-newton-schulz).
The port retains ANVIL's BF16 and six coefficients, not the reference package's
FP16/default maps. The fused polynomial includes its diagonal shift before one
BF16 store. Products are mirrored from the lower triangle. Floating-point
updates differ from baseline; equivalent algebra does not prove equal loss.
In a full-size dense-reference probe, the relative L2 update difference was
about 1.1% for ordinary inputs and 36.6% for an artificial rank-8 input. Outputs
remained finite and spectrally bounded. This makes the convergence check essential;
these numbers are not loss changes or results from the production GPU pipeline.

| Setting | Behavior |
| --- | --- |
| `ANVIL_MLP_WHITENING=gram_pair` | Default on this branch; MLP experiment |
| `ANVIL_MLP_WHITENING=standard` | Original six-pass computation; comparison control |
| W&B name | `gram-ns-mlp-1gpu-seed-999` (or the selected seed) |
| Control W&B name | `baseline-1gpu-seed-999` |

Both trainers log the mode. The offline uploader records it in W&B configuration.
Set the mode before launching a process; it is fixed during graph capture.

## Validate on the cluster

Inside an existing one-5090 allocation:

```bash
source ~/venvs/nanogpt/bin/activate
cd ~/speed-run/modded-nanogpt
git fetch origin
git switch --track origin/experiment/gram-ns-mlp
python -u tests/check_gram_ns.py && python -u tests/check_rtx5090.py --mlp-whitening gram_pair
```

If the branch already exists, use `git switch experiment/gram-ns-mlp` and
`git pull --ff-only origin experiment/gram-ns-mlp` instead of `--track`.
Do not discard a dirty checkout. The first test needs no dataset. It checks the
new kernel, full-size numerical stress cases, frozen slots, optimizer state,
CUDA graph replay after reset/scalar changes, and interleaved bank timing.
It prints `GRAM_NS_CHECK status=pass` only after all checks finish.

Batch 24 represents one RTX rank. Batch 3 represents an H100 rank's local matrix
count, without testing collectives. Neither result proves eight-GPU speedup.
The second test checks all-stage warmup, two training updates and validation.
Logs: `logs/check-gram-ns-*.txt` and `logs/check-rtx5090-*.txt`.

After both pass, compare full seed-999 runs in `standard` and `gram_pair` modes
on the same GPU/settings. Use the normal 1194-update schedule. Check final
validation loss and total recorded time; bank timings alone do not promote it.
Expand to five seeds only if the pair is promising. Validate on 8×H100 separately.

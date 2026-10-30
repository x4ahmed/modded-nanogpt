# RTX 5090 diagnostics

## Current experiment: MLP Gram Newton–Schulz

Run `python -u tests/check_gram_ns.py`, then
`python -u tests/check_rtx5090.py --mlp-whitening gram_pair` if it passes.
The first test checks kernel parity, numerical stress cases, optimizer graph
replay/reset, frozen slots and complete bank-update timings for 3 and 24 matrices.
It needs no dataset and writes `logs/check-gram-ns-*.txt` immediately.
Three matrices simulate a rank's local H100 workload, not eight-GPU communication.
`--mlp-whitening standard` selects the unchanged control in the training smoke test.
See [experiment settings and interpretation](../docs/gram-ns-mlp.md).

## Port diagnostics

Inside a Slurm allocation with one RTX 5090 and the training venv active:

```bash
python tests/diagnose_rtx5090.py
```

The nine cases run sequentially in fresh processes. Failures do not stop later
cases. Compilation can take tens of minutes. Each case has a 30-minute timeout.
Leave the terminal connected. Do not run another GPU test at the same time.

| Case | Question |
| --- | --- |
| gemm | Does FP8 matrix multiplication fail for a particular output dtype or fast-accumulation setting? |
| mlp | Do repeated forward and independent backward MLP kernels produce finite outputs, matching transposes and sampled reference rows, eagerly and compiled? |
| ce | Does the CE kernel work on known finite logits at every candidate width? |
| eval | Is the fresh BF16 evaluation forward finite? |
| sampled | Does the unmodified compiled training forward/backward work with sampled softmax? |
| full | Does the same test work with full-vocabulary softmax? |
| bf16_mlp_control | Does replacing only MLP arithmetic remove the NaN? |
| bf16_logits_control | Does BF16 GEMM output followed by FP8 conversion remove the NaN? |
| trace | Which attention, MLP, head-input or logit boundary first becomes non-finite? |

The controls are temporary changes inside test processes, not proposed fixes.
Trace uses segmented compilation, so a trace-only failure may be instrumentation
related. Compare its result with the unmodified compiled cases. Evaluation and
training losses use different objectives; their values need not match.

Default input: seed 999, 16,384 training tokens, n-gram divisor 32. The model uses
65,536-token evaluation capacity, but each forward case checks one training-size
batch. This is not the full validation benchmark or a memory preflight. GEMM/MLP
parity checks sample rows and use loose FP8 screening tolerances, not bitwise parity.

Report the final summary and the entire new `logs/rtx-diagnostics-*/` directory.
It contains one log per case and `summary.json`. Exact input and initialization
sample fingerprints are printed for comparing the fresh-model cases.

Optional: `--case trace` reruns one case; `--tokens 2048` tests a smaller shape;
`--timeout 3600` allows an hour per case. Start with the default suite.

CPU runner checks: `python -m unittest discover -s tests -p test_rtx_diagnostic_runner.py`.

After diagnostics pass, run `python -u tests/check_rtx5090.py`. This exercises
all-stage warmup, two optimizer updates, and validation. Do not launch baseline
training until this check also passes. These tests are not baseline timing runs.

RTX narrow-tile MLP outputs now use masked pointer stores in both directions.
The FP8 math, scales, shapes and model structure are unchanged. This bypasses
the suspect descriptor-store epilogue; GPU results must verify the repair.
CPU routing/coverage checks: `python -m unittest discover -s tests -p test_mlp_store_contract.py`.

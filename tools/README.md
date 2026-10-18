# Post-training W&B upload

Training stays unchanged: no W&B import, background process or network upload.
The launcher now calls the uploader automatically after successful training exits,
targeting `i-learn/speedrun`. It captures `logs/run-<timestamp>-<pid>.console.txt`.
It uses `~/venvs/wandb-upload/bin/python`; override with `WANDB_UPLOAD_PYTHON`.
Preflight and failed training runs are not uploaded. `RTX_WANDB_UPLOAD=0` disables
automatic uploads. Successful truncated runs are uploaded as `full_schedule=false`.
Upload failures print a retry command and preserve training success and local logs.
Uploading prolongs the job/allocation lifetime, but is outside measured training time.
Direct Python training/checker invocations do not auto-upload; use `bash run_rtx.sh`.
The uploader needs no GPU and works on a login
node (if cluster policy permits) or a PC with a copied log and internet access.

One-time setup, preferably in a separate environment so training dependencies stay fixed:

```bash
python3 -m venv ~/venvs/wandb-upload
source ~/venvs/wandb-upload/bin/activate
python -m pip install wandb
wandb login
```

Follow the login prompt to obtain/paste your API key. Do not put the key in a
command argument, repository, job script, log, or chat. The CLI stores credentials
in your user account. For temporary credentials instead, use Bash's hidden input:

```bash
read -rsp 'W&B API key: ' WANDB_API_KEY; echo
export WANDB_API_KEY
# Run the upload, then:
unset WANDB_API_KEY
```

From the repository, replace the filename with ONE actual completed run log:

```bash
python tools/upload_wandb.py logs/ACTUAL_LOG.txt --dry-run
python tools/upload_wandb.py logs/ACTUAL_LOG.txt --name pilot-1gpu-seed-999 --seed 999
```

Use either the trainer UUID log or console log, not both. Add `--entity TEAM` if
needed. Optional `--training-commit HASH` must be the commit used for training;
the uploader deliberately does not use the current Git checkout or environment.

Uploads: loss history at existing logging intervals, recorded training time,
final validation loss, peak memory, configuration from the runtime header,
and a canonical-mask fallback flag. No raw logs, source snapshots or checkpoints
are attached. Uploader system metrics/code/Git collection are disabled. W&B's
run duration is upload duration: compare `training_time_ms`, not W&B `_runtime`.
Unlogged metrics (e.g. historical GPU utilization or LR) cannot be reconstructed.
Training loss includes the scheduled auxiliary objectives; do not equate it to
validation cross-entropy. Timing is preserved, including mask-building overhead.

Incomplete/multi-run/nonfinite logs are rejected. Completed smoke tests can be
imported but are marked `full_schedule=false`; successful import is not a claim
of convergence. The same file maps to a stable run ID; `resume=never` refuses an
existing ID to prevent accidental duplicate history. If a network upload fails,
keep W&B's local files and use the SDK's suggested `wandb sync` recovery command
rather than changing the source log to force a fresh ID.

Reference: https://docs.wandb.ai/ref/python/experiments/run/

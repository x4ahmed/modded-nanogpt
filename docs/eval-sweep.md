# Evaluation sweep: free loss at validation time

Branch `experiment/eval-sweep`, from `baseline` (5be1ba0). Training and the official final validation are
unchanged; the official `final_val_loss` and training time are the baseline's. After the final summary line,
the trainer re-scores the same final weights under evaluation-time variants. Validation is untimed both
here and on 8xH100, so any variant that lowers the loss costs no training time.

## What is swept

- **Attention windows.** The record widens only the two long-window layers for the final validation
  (13 -> 20 blocks, attention mask only; `apply_final_ws_ext`). The five short-window layers stay at the
  extension stage's 6 blocks. The sweep scores short windows +0, +2, +4, +6 blocks and, separately, the
  long window +4 blocks, all mask only, as the record does for its long windows.
- **Softmax temperature.** Each window pair is scored at temperatures 0.90, 0.95, 0.98, 1.00, 1.02, 1.05,
  1.10 from one forward pass (`model.eval_temperatures`; `cross_entropy_at_temperatures` in
  `model/layers.py`). Training mixes in auxiliary objectives early, uses a sampled softmax for most steps
  and averages weights at the end; a temperature away from 1.00 would mean the final model is
  miscalibrated. The canonical mask's dropped tokens stay at -60 / temperature, still negligible.

Every variant runs on the same weights, so differences carry no seed noise: a 0.0002 change is real.
The first window pair at 1.00 is the official setting; the sweep checks that it reproduces
`final_val_loss` (within 1e-4) and prints `ok` or `MISMATCH`.

## Output

```
EVAL_SWEEP ws_short=6 ws_long=20 temperature=1.00 val_loss=...
...
EVAL_SWEEP official=... reproduced=... ok
```

Five window pairs, about 10-15 extra minutes after a full run (compilation per pair included). Smoke
tests (short runs) sweep only the first two pairs on two validation chunks, to exercise the code.
W&B runs from this branch are `eval-sweep-1gpu-seed-<seed>`; their official numbers are baseline runs.

## Reading it

A variant is worth adopting if it beats the official setting on this run and the direction repeats on a
second seed's run. Adopting it means changing evaluation constants only. On 8xH100 each 0.001 of loss is
worth ~0.16 s at the record's rate.

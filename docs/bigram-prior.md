# Bigram output prior (awaiting GPU validation)

Branch `experiment/bigram-prior`, created from `baseline` (`5be1ba0`). `RTX_BIGRAM_PRIOR=1` enables it;
the default `0` is the unchanged baseline. Code: `track_1_short/bigram_prior.py`.

## What it adds

For the previous token a and each of its K most frequent continuations b:

    val(a, b) = log p(b | a) - log p(b),    p(b | a) = (c(a, b) + kappa * p(b)) / (c(a) + kappa)

with p(b) the add-one-smoothed unigram. Every loss path (sampled training CE, full training CE,
validation) adds `beta * val(input_token, b)` to the raw logit of class b, before the softcap.
Other classes get 0 ("no evidence beyond the unigram"). `beta` is one learned scalar, init 1.0
(replicated Adam, base LR 0.008 × 1.0, betas 0.9/0.99, no weight decay).

| Setting | Default | Env |
| --- | --- | --- |
| Tokens counted | 100,000,000 (first training shard(s), sorted order) | `RTX_BIGRAM_TOKENS` |
| Continuations per previous token | 64 | `RTX_BIGRAM_K` |
| Dirichlet smoothing kappa | 1000 | `RTX_BIGRAM_KAPPA` |

## Constraints it keeps

- **No leakage.** Counts come only from training shards: the first 100M tokens of the stream the run
  then trains on (it consumes roughly 300M+). Validation data is never read.
- **Smoothed.** Unseen continuations keep the unigram backoff (value 0), never zero probability.
- **On the clock.** The build (shard read, GPU count, top-K) runs after the timer starts, like the
  prefix table; the log line `BIGRAM_PRIOR ... build_ms=` reports its share.

## Known caveats

- Product-of-experts, not interpolation: the prior is added to logits. The softcap compresses large
  values, so `beta`'s learned value is part of the result.
- The same logit row also serves the multi-token and prefix-token targets, which a next-token bigram
  prior does not fit; `beta` trades this off.
- Training logits are e4m3: the prior is added in fp32 and rounded once more to e4m3 (one extra rounding).
- Class 0 is the empty-slot marker and never gets a prior; in the sampled phase the candidate at
  position 0 does not either. Both are one class out of 50,257.
- Continuations outside the top 64 get 0, slightly favouring them relative to the true smoothed model.
- Extra cost per microbatch: a [T, 64] gather/scatter on the logits and one more in the backward.

## Commands

GPU smoke: `python -u tests/check_rtx5090.py --bigram-prior 1 --seed 999` (must print `RTX5090_CHECK
status=pass`). Then five seeds with `RTX_BIGRAM_PRIOR=1`; W&B names `bigram-prior-1gpu-seed-<seed>`.
Compare paired with the baseline (loss 3.2948572 ± 0.00104874, time 861.635 ± 1.608645 s at `5be1ba0`).

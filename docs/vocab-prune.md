# Vocabulary pruning (Phase 2, idea 2)

Branch: `experiment/vocab-prune`, directly from frozen `baseline` (`5be1ba0`). Idea (Notion, 2026-11-05):
check how often each token appears in the training data and drop the tokens that are not needed.
Status: analysis. The training code is the baseline's, unchanged; only the census tool is new.

## Where the vocabulary still costs time

The record already prunes the vocabulary twice:

- Sampled softmax (record #360, `track_1_short/sampled_softmax.py`): steps 0-1106 normalize the training
  loss over P = 10,240 / 14,336 / 24,576 candidates (every target of the microbatch plus uniform stride
  negatives), not over 50,304 ids. The width is P, so a smaller vocabulary does not change these steps.
- Canonical mask (record #350, `track_1_short/canonical_mask.py`): validation drops ids the GPT-2
  tokenizer can never emit after the previous token.

| Vocabulary-sized work | When | Gain from a smaller vocabulary |
| --- | --- | --- |
| lm_head GEMMs and CE, full softmax | steps 1107-1193 (87 steps, 24.6M of 328.7M training tokens, 7.5%) | proportional to the ids removed |
| lm_head GEMMs and CE, sampled | steps 0-1106 | none (width P) |
| embed / value-embedding lookups | every step | none (gathers) |
| Adam on lm_head, embed, 4 value-embedding planes (6 x V x 768) | update steps | small (memory-bound) |
| validation forward | after training | none: off the clock |

Size of the remaining lever, from our own measurement: `record_late` (2026-10-28, same allocation as its
control) trained the 67 taper steps at P = 24,576 instead of 50,304 and saved 1.7 s, about 1 ms per
1,000 output ids per batch-20 step. Over the 87 full-softmax steps that is ~0.1 s per 1,000 ids removed;
a FLOP bound (24.6M tokens x 768 x 1,000 ids x 6) gives at most ~0.4 s. Same-allocation pair noise in
time is ~1-2 s, so pruning must remove several thousand ids before a pair can see it.

Out of scope: retokenizing with a smaller BPE (changes the token stream and the loss-per-GPT-2-token
metric) and choosing ids from validation counts (peeking). Any id removed from the output must keep a
finite probability: under the -60 mask a validation token on it costs ~80 nats.

## Step 1: census

`tools/token_census.py` replays exactly the token stream the one-GPU run trains on (the loader's
BOS-aligned slicing, per-stage document cuts, 16,384-token microbatches, 1194 steps, 328,663,040 target
tokens) and counts every id as a target and as an input; it also counts the 10,485,760 validation
targets, only to measure the risk of a train-derived choice. CPU only, no training, no GPU needed.

```bash
source ~/venvs/nanogpt/bin/activate
cd ~/speed-run/modded-nanogpt
git fetch origin
git switch experiment/vocab-prune
git pull --ff-only origin experiment/vocab-prune
python -u tools/token_census.py
```

It prints a short report (ids that occur, ids needed for 99-99.999% coverage, and for train counts
below 1 / 10 / 100 / 1000: how many ids, their share of training targets, and how many validation
tokens fall on them), then `CENSUS_COMPLETE dir=logs/token-census/<UTC>`. The folder keeps
`counts.npz` (per-id counts) and `summary.json`. CPU tests: `tests/test_token_census.py` (slicing
identical to the record's `Shard` on synthetic shards, counts against brute force, schedule totals).

## Decision after the census

- Fewer than ~5,000 removable ids (train count 0, validation hits ~0): the time gain is below pair noise.
  Close the idea and record the numbers.
- More: two candidates, each its own paired run (`tools/run_pair.py` from `experiment/attnres-last-init`):
  - A. Pruned output in the full-softmax steps: run steps 1107-1193 through the sampled-softmax path
    with a fixed candidate set = ids seen as training targets so far (every target is in it by
    construction); negatives only from seen ids. Validation keeps all 50,304 ids (metric unchanged).
  - B. Eval-time mask of never-seen ids, an extension of the canonical mask, counted on the fly from
    the run's own targets. No training cost; measured exactly inside one run by computing the
    validation loss with and without the extra mask from the same forward. Expected gain is tiny:
    those ids already sit near the softcap floor.

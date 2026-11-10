"""Token census for the Phase 2 idea "prune the vocabulary": how often each GPT-2 token id occurs in
exactly the tokens the one-GPU run trains on, and in the validation tokens it is scored on.

The training stream is replayed as portable/data.py's MicrobatchLoader consumes it: train shards in
sorted order; per step, batch_size / RTX_MICROBATCH_TOKENS slices cut by data.Shard.next_batch's rule
(BOS-aligned documents, each cut to the stage's train_max_seq_len, the last one cut to fill the slice);
a shard's unused tail is skipped when a step no longer fits. Slice targets are tokens[1:], inputs
tokens[:-1]. Validation targets are the first val_tokens + 1 tokens of the first val shard, minus the
first (validation_batches). The replay uses the full BOS index; the loader's partial first-6M-token
index is a prefix of it, so the slices are identical.

Validation counts measure the risk of a train-derived choice only; never pick tokens from them.

CPU only: numpy, plus torch for the schedule; no GPU and no training. Writes
logs/token-census/<UTC>/counts.npz and summary.json; prints CENSUS_COMPLETE.
"""
import argparse
from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from track_1_short.config import (  # noqa: E402
    LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, TRAINING_STAGES, WS_POST_YARN_EXT, Hyperparameters,
)
from track_1_short.schedule import TrainingSchedule  # noqa: E402

MAGIC, VERSION, HEADER_BYTES = 20240520, 1, 256 * 4
BOS_ID = 50256           # track_1_short/data.py
REAL_VOCAB = 50257       # GPT-2 ids; the model pads to 50304, the padding ids never occur
PADDED_VOCAB = 50304
FULL_SOFTMAX_START = 1107  # sampled_softmax.py: steps 0-1106 train on 10240-24576 candidates
THRESHOLDS = (1, 10, 100, 1000)
COVERAGE = (0.99, 0.999, 0.9999, 0.99999)


def read_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    if header.size != 256 or header[0] != MAGIC or header[1] != VERSION:
        raise ValueError(f"{path}: not a version-1 token shard")
    tokens = np.fromfile(path, dtype=np.uint16, count=int(header[2]), offset=HEADER_BYTES)
    if tokens.size != header[2]:
        raise ValueError(f"{path}: header claims {header[2]} tokens, read {tokens.size}")
    return tokens


class ShardCursor:
    """data.Shard.next_batch on numpy, one slice per microbatch."""

    def __init__(self, tokens):
        self.tokens, self.size, self.i = tokens, tokens.size, 0
        self.bos = np.flatnonzero(tokens == BOS_ID)

    def next_batch(self, slices, slice_tokens, max_seq_len):
        """(starts, ends) per slice, each slice_tokens + 1 tokens; StopIteration at the shard's tail."""
        bos, n, idx, out = self.bos, len(self.bos), self.i, []
        for _ in range(slices):
            starts, ends, cur_len = [], [], 0
            while cur_len <= slice_tokens:
                if idx >= n:
                    raise StopIteration("Insufficient BOS ahead; hit tail of shard.")
                cur = int(bos[idx])
                idx += 1
                end = min(int(bos[idx]) if idx < n else self.size, cur + max_seq_len,
                          cur + slice_tokens - cur_len + 1)
                starts.append(cur)
                ends.append(end)
                cur_len += end - cur
            assert cur_len == slice_tokens + 1
            out.append((starts, ends))
        self.i = idx
        return out


def replay_training(files, schedule, microbatch_tokens, steps=None):
    """Per-id counts of the training targets and inputs; targets also split at FULL_SOFTMAX_START."""
    counts = {k: np.zeros(PADDED_VOCAB, np.int64) for k in ("targets", "inputs", "targets_full_softmax")}
    file_iter = iter(files)
    cursor, shards = ShardCursor(read_shard(next(file_iter))), 1
    for step in range(schedule.total_steps if steps is None else steps):
        stage, _ = schedule.lookup(step)
        assert stage.batch_size % microbatch_tokens == 0
        while True:
            try:
                slices = cursor.next_batch(stage.batch_size // microbatch_tokens, microbatch_tokens,
                                           stage.train_max_seq_len)
                break
            except StopIteration:
                path = next(file_iter, None)
                if path is None:
                    raise RuntimeError(f"out of training data at step {step}") from None
                cursor, shards = ShardCursor(read_shard(path)), shards + 1
        tok = cursor.tokens
        seen = np.bincount(np.concatenate([tok[s:e] for st, en in slices for s, e in zip(st, en)]),
                           minlength=PADDED_VOCAB)
        first = np.bincount(tok[[st[0] for st, _ in slices]], minlength=PADDED_VOCAB)
        last = np.bincount(tok[[en[-1] - 1 for _, en in slices]], minlength=PADDED_VOCAB)
        counts["targets"] += seen - first
        counts["inputs"] += seen - last
        if step >= FULL_SOFTMAX_START:
            counts["targets_full_softmax"] += seen - first
    return counts, shards


def count_validation(files, val_tokens):
    tokens = read_shard(files[0])
    if tokens.size <= val_tokens + 1:
        raise RuntimeError("first validation shard is shorter than the validation stream")
    return (np.bincount(tokens[1:val_tokens + 1], minlength=PADDED_VOCAB),
            np.bincount(tokens[:val_tokens], minlength=PADDED_VOCAB))


def decoder():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return lambda i: enc.decode_single_token_bytes(int(i)).decode("utf-8", "backslashreplace")
    except Exception:  # offline or not installed: ids only
        return None


def summarize(train, val, decode=None):
    t, v = train[:REAL_VOCAB], val[:REAL_VOCAB]
    assert not train[REAL_VOCAB:].any() and not val[REAL_VOCAB:].any(), "padding ids occurred"
    order = np.argsort(-t, kind="stable")
    share = np.cumsum(t[order]) / t.sum()
    rare = {}
    for c in THRESHOLDS:
        mask = t < c
        rare[f"train_count_below_{c}"] = dict(
            ids=int(mask.sum()), kept_vocab=int(REAL_VOCAB - mask.sum()),
            train_target_share=float(t[mask].sum() / t.sum()),
            val_tokens_hit=int(v[mask].sum()), val_ids_hit=int((mask & (v > 0)).sum()))
    unseen = np.flatnonzero(t == 0)
    unseen_in_val = unseen[np.argsort(-v[unseen], kind="stable")]
    unseen_in_val = unseen_in_val[v[unseen_in_val] > 0][:20]
    label = (lambda i: dict(id=int(i), text=decode(i))) if decode else (lambda i: dict(id=int(i)))
    return dict(
        train_target_tokens=int(t.sum()), val_target_tokens=int(v.sum()),
        distinct_train_targets=int((t > 0).sum()), distinct_val_targets=int((v > 0).sum()),
        ids_for_train_coverage={str(q): int(np.searchsorted(share, q) + 1) for q in COVERAGE},
        rare=rare,
        never_seen_examples=[label(i) for i in unseen[:30]],
        never_seen_but_in_val=[dict(label(i), val_count=int(v[i])) for i in unseen_in_val],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--microbatch-tokens", type=int,
                        default=int(os.environ.get("RTX_MICROBATCH_TOKENS", "16384")))
    parser.add_argument("--out", type=Path, default=None, help="default: logs/token-census/<UTC>")
    cli = parser.parse_args()
    if "NUM_SCHEDULED_ITERATIONS" in os.environ:
        parser.error("unset NUM_SCHEDULED_ITERATIONS: the census replays the full 1194-step run")
    args = Hyperparameters()
    schedule = TrainingSchedule(TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations,
                                device="cpu", cooldown_frac=LR_COOLDOWN_FRAC, split_embed_stage=SPLIT_EMBED_STAGE,
                                ws_post_yarn_ext=WS_POST_YARN_EXT)
    train_files, val_files = sorted(glob.glob(args.train_files)), sorted(glob.glob(args.val_files))
    if not train_files or not val_files:
        raise SystemExit(f"no shards at {args.train_files} / {args.val_files} (set DATA_PATH?)")
    print(f"Replaying {schedule.total_steps} steps at {cli.microbatch_tokens} tokens per microbatch...", flush=True)
    counts, shards = replay_training(train_files, schedule, cli.microbatch_tokens)
    expected = sum(schedule.lookup(s)[0].batch_size for s in range(schedule.total_steps))
    assert counts["targets"].sum() == expected, (counts["targets"].sum(), expected)
    val_targets, val_inputs = count_validation(val_files, args.val_tokens)
    summary = summarize(counts["targets"], val_targets, decoder())
    summary.update(steps=schedule.total_steps, microbatch_tokens=cli.microbatch_tokens, train_shards_read=shards,
                   full_softmax_target_tokens=int(counts["targets_full_softmax"].sum()),
                   distinct_full_softmax_targets=int((counts["targets_full_softmax"] > 0).sum()),
                   never_seen_as_input=int((counts["inputs"][:REAL_VOCAB] == 0).sum()))
    out = cli.out or ROOT / "logs" / "token-census" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "counts.npz", train_targets=counts["targets"], train_inputs=counts["inputs"],
                        train_targets_full_softmax=counts["targets_full_softmax"],
                        val_targets=val_targets, val_inputs=val_inputs)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n".join(report(summary)))
    print(f"CENSUS_COMPLETE dir={out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}", flush=True)


def report(summary):
    """The summary in a few pasteable lines (summary.json has the same numbers)."""
    def token(entry):
        return f"{entry['id']}{entry['text']!r}" if "text" in entry else str(entry["id"])
    lines = [
        f"Training targets: {summary['train_target_tokens']:,} tokens, {summary['steps']} steps, "
        f"{summary['train_shards_read']} shards read. {summary['distinct_train_targets']:,} of {REAL_VOCAB:,} ids "
        f"occur ({summary['distinct_full_softmax_targets']:,} in the full-softmax steps); "
        f"{summary['never_seen_as_input']:,} ids never occur as inputs.",
        "Ids covering the training targets: "
        + ", ".join(f"{float(q) * 100:g}% -> {k:,}" for q, k in summary["ids_for_train_coverage"].items()),
    ]
    for c in THRESHOLDS:
        x = summary["rare"][f"train_count_below_{c}"]
        lines.append(f"Train count < {c:>4}: {x['ids']:>6,} ids (keep {x['kept_vocab']:,}), "
                     f"{x['train_target_share']:.2e} of training targets; validation tokens on them: "
                     f"{x['val_tokens_hit']:,} of {summary['val_target_tokens']:,} ({x['val_ids_hit']:,} ids)")
    lines.append("Never in training (first ids): " + ", ".join(map(token, summary["never_seen_examples"])))
    lines.append("Never in training but in validation (most frequent): "
                 + (", ".join(f"{token(e)} x{e['val_count']}" for e in summary["never_seen_but_in_val"]) or "none"))
    return lines


if __name__ == "__main__":
    main()

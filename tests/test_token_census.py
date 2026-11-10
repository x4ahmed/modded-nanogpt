import ast
import importlib.util
from pathlib import Path
import re
import tempfile
import threading
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("token_census", ROOT / "tools/token_census.py")
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)


def record_shard_class():
    """data.Shard, compiled from the record's source (data.py's other imports need triton/CUDA)."""
    tree = ast.parse((ROOT / "track_1_short/data.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Shard")
    namespace = dict(torch=torch, threading=threading, np=np, BOS_ID=census.BOS_ID, Tensor=Tensor, Path=Path)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "data.py", "exec"), namespace)
    return namespace["Shard"]


def synthetic_tokens(rng, docs, max_len=700):
    parts = []
    for _ in range(docs):
        parts += [[census.BOS_ID], rng.integers(0, 3000, rng.integers(1, max_len)).tolist()]
    return np.array([t for p in parts for t in p], dtype=np.uint16)


def write_shard(path, tokens, magic=census.MAGIC):
    header = np.zeros(256, dtype=np.int32)
    header[:3] = magic, census.VERSION, tokens.size
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


class FakeSchedule:
    def __init__(self, stages):
        self.stages, self.total_steps = stages, len(stages)

    def lookup(self, step):
        batch_size, max_seq_len = self.stages[step]
        return SimpleNamespace(batch_size=batch_size, train_max_seq_len=max_seq_len), 0.0


class CensusTests(unittest.TestCase):
    def test_slicing_matches_the_record_shard(self):
        Shard, rng = record_shard_class(), np.random.default_rng(0)
        tokens = synthetic_tokens(rng, 400)
        record = Shard(torch.from_numpy(tokens.astype(np.int32)))
        record._ready.wait()
        mine = census.ShardCursor(tokens)
        calls = 0
        while True:
            slices, mb, seq = int(rng.integers(1, 4)), 256 * int(rng.integers(1, 4)), int(rng.choice([96, 300, 900]))
            record.world_size = slices
            try:
                want = record.next_batch(mb, seq)
            except StopIteration:
                with self.assertRaises(StopIteration):
                    mine.next_batch(slices, mb, seq)
                break
            got = mine.next_batch(slices, mb, seq)
            self.assertEqual([(list(map(int, s)), list(map(int, e))) for s, e in zip(*want)], got)
            calls += 1
        self.assertGreater(calls, 20)

    def test_replay_counts_targets_and_inputs_across_shards(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as folder:
            files = []
            for i in range(8):
                files.append(Path(folder) / f"train_{i:06d}.bin")
                write_shard(files[-1], synthetic_tokens(rng, 12))
            stages = [(1024, 300)] * 3 + [(2048, 900)] * 4
            census_stop = census.FULL_SOFTMAX_START
            census.FULL_SOFTMAX_START = 5
            try:
                counts, shards = census.replay_training(files, FakeSchedule(stages), 512)
            finally:
                census.FULL_SOFTMAX_START = census_stop
            # Brute force: the same slices, concatenated per microbatch, shifted by one.
            targets, inputs, late = (np.zeros(census.PADDED_VOCAB, np.int64) for _ in range(3))
            it = iter(files)
            cursor = census.ShardCursor(census.read_shard(next(it)))
            for step, (batch, seq) in enumerate(stages):
                while True:
                    try:
                        slices = cursor.next_batch(batch // 512, 512, seq)
                        break
                    except StopIteration:
                        cursor = census.ShardCursor(census.read_shard(next(it)))
                for st, en in slices:
                    buf = np.concatenate([cursor.tokens[s:e] for s, e in zip(st, en)])
                    self.assertEqual(buf.size, 513)
                    targets += np.bincount(buf[1:], minlength=census.PADDED_VOCAB)
                    inputs += np.bincount(buf[:-1], minlength=census.PADDED_VOCAB)
                    if step >= 5:
                        late += np.bincount(buf[1:], minlength=census.PADDED_VOCAB)
        self.assertGreater(shards, 1)
        np.testing.assert_array_equal(counts["targets"], targets)
        np.testing.assert_array_equal(counts["inputs"], inputs)
        np.testing.assert_array_equal(counts["targets_full_softmax"], late)
        self.assertEqual(counts["targets"].sum(), sum(b for b, _ in stages))

    def test_out_of_data_and_bad_shards_fail_loudly(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train.bin"
            write_shard(path, synthetic_tokens(np.random.default_rng(2), 5, max_len=50))
            with self.assertRaises(RuntimeError):
                census.replay_training([path], FakeSchedule([(4096, 900)]), 4096)
            bad = Path(folder) / "bad.bin"
            write_shard(bad, np.arange(10), magic=1)
            with self.assertRaises(ValueError):
                census.read_shard(bad)

    def test_validation_counts_the_scored_stream(self):
        tokens = np.arange(100, dtype=np.uint16) % 7
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "val.bin"
            write_shard(path, tokens)
            targets, inputs = census.count_validation([path], 64)
            with self.assertRaises(RuntimeError):
                census.count_validation([path], 99)
        np.testing.assert_array_equal(targets, np.bincount(tokens[1:65], minlength=census.PADDED_VOCAB))
        np.testing.assert_array_equal(inputs, np.bincount(tokens[:64], minlength=census.PADDED_VOCAB))

    def test_summary_thresholds_coverage_and_val_hits(self):
        train = np.zeros(census.PADDED_VOCAB, np.int64)
        train[:4] = 1000, 100, 10, 1   # ids 4.. never seen
        val = np.zeros(census.PADDED_VOCAB, np.int64)
        val[[0, 3, 7, 9]] = 50, 2, 3, 1
        s = census.summarize(train, val)
        self.assertEqual(s["distinct_train_targets"], 4)
        self.assertEqual(s["rare"]["train_count_below_1"],
                         dict(ids=census.REAL_VOCAB - 4, kept_vocab=4, train_target_share=0.0,
                              val_tokens_hit=4, val_ids_hit=2))
        self.assertEqual(s["rare"]["train_count_below_10"]["ids"], census.REAL_VOCAB - 3)
        self.assertEqual(s["rare"]["train_count_below_10"]["val_tokens_hit"], 6)
        # cumulative shares 0.9001, 0.9901, 0.9991, 1.0
        self.assertEqual(s["ids_for_train_coverage"], {"0.99": 2, "0.999": 3, "0.9999": 4, "0.99999": 4})
        self.assertEqual([e["id"] for e in s["never_seen_but_in_val"]], [7, 9])
        train[census.REAL_VOCAB] = 1
        with self.assertRaises(AssertionError):
            census.summarize(train, val)

    def test_constants_match_the_record(self):
        source = (ROOT / "track_1_short/data.py").read_text(encoding="utf-8")
        self.assertEqual(int(re.search(r"^BOS_ID = (\d+)", source, re.M).group(1)), census.BOS_ID)
        tree = ast.parse((ROOT / "track_1_short/sampled_softmax.py").read_text(encoding="utf-8"))
        counts = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                      and getattr(n.targets[0], "id", "") == "CANDIDATES_BY_STAGE")
        self.assertEqual(sorted(counts), [0, 1, 2])  # stages 3 and 4 run the full softmax
        args = census.Hyperparameters()
        schedule = census.TrainingSchedule(
            census.TRAINING_STAGES, 1122, args.num_extension_iterations, device="cpu",
            cooldown_frac=census.LR_COOLDOWN_FRAC, split_embed_stage=census.SPLIT_EMBED_STAGE,
            ws_post_yarn_ext=census.WS_POST_YARN_EXT)
        self.assertEqual(schedule.total_steps, 1194)
        self.assertEqual(schedule.boundaries[3][0], census.FULL_SOFTMAX_START)
        self.assertEqual(sum(schedule.lookup(s)[0].batch_size for s in range(1194)), 328_663_040)


if __name__ == "__main__":
    unittest.main()

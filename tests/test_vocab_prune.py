"""CPU tests of the dynamic vocabulary pruning (track_1_short/portable/vocab_prune.py and its wiring in
portable/sampled.py). The Triton transpose kernel is replaced by a torch transpose; GPU kernels are not run."""
import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from track_1_short.config import LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, TRAINING_STAGES, WS_POST_YARN_EXT
from track_1_short.portable import vocab_prune
from track_1_short.portable.vocab_prune import (
    KEPT_IDS_BY_MASS, MIN_RANKED_WIDTH, PRUNED_WIDTH, RankedCandidates, SeenTargets, build_candidate_set,
    pruned_counts, ranked_counts, ranked_width,
)
from track_1_short.schedule import TrainingSchedule

CENSUS_DISTINCT_TARGETS = 49_989  # tools/token_census.py on the cluster, 2026-11-10
VOCAB = 50_304


def import_portable_sampled():
    stub = types.ModuleType("track_1_short.perf.kernels.transpose")
    stub.transpose_copy = lambda src, dst: dst.copy_(src.T)
    with patch.dict(sys.modules, {"track_1_short.perf.kernels.transpose": stub}):
        for name in ("track_1_short.sampled_softmax", "track_1_short.portable.sampled"):
            sys.modules.pop(name, None)
        from track_1_short.portable import sampled
        return sampled


def schedule():
    return TrainingSchedule(TRAINING_STAGES, 1122, 20, device="cpu", cooldown_frac=LR_COOLDOWN_FRAC,
                            split_embed_stage=SPLIT_EMBED_STAGE, ws_post_yarn_ext=WS_POST_YARN_EXT)


class CandidateSetTests(unittest.TestCase):
    def test_width_is_the_largest_exact_cut(self):
        self.assertEqual(PRUNED_WIDTH % 128, 0)
        self.assertEqual(PRUNED_WIDTH, VOCAB - 2 * 128)
        self.assertGreaterEqual(PRUNED_WIDTH, CENSUS_DISTINCT_TARGETS)
        self.assertLess(PRUNED_WIDTH - 128, CENSUS_DISTINCT_TARGETS)

    def test_set_holds_every_seen_id_and_drops_the_highest_unseen(self):
        seen = np.zeros(20, dtype=bool)
        seen[[0, 3, 4, 17]] = True
        cand = build_candidate_set(seen, 8)
        np.testing.assert_array_equal(cand, [0, 1, 2, 3, 4, 5, 6, 17])
        self.assertIsNone(build_candidate_set(seen, 3))
        np.testing.assert_array_equal(build_candidate_set(seen, 4), [0, 3, 4, 17])

    def test_full_softmax_steps_become_pruned(self):
        self.assertEqual(pruned_counts([10240, 0, 24576, 0]), [10240, PRUNED_WIDTH, 24576, PRUNED_WIDTH])

    def test_refresh_states(self):
        seen = SeenTargets(20, 8, "cpu")
        seen.mark(torch.tensor([1, 2, 2]))
        self.assertEqual(seen.refresh(np.array([1, 2, 2])), "new")
        np.testing.assert_array_equal(seen.candidates, np.arange(8))  # 1, 2 seen; 0, 3-7 fill
        self.assertEqual(seen.refresh(np.array([2, 1])), "same")
        seen.mark(torch.tensor([15]))  # a target outside the set: rebuilt around it, still 8 wide
        self.assertEqual(seen.refresh(np.array([15, 1])), "new")
        self.assertIn(15, seen.candidates)
        self.assertEqual(len(seen.candidates), 8)
        seen.mark(torch.arange(10, 19))  # 1, 2 and 10-18: 11 seen > 8
        self.assertEqual(seen.refresh(np.array([18])), "full")
        self.assertEqual(seen.stats["full_softmax_microbatches"], 1)
        self.assertEqual(seen.stats["builds"], 2)
        self.assertIn("seen_total=11", seen.summary())
        seen.reset()
        self.assertFalse(seen.seen.any())
        self.assertIsNone(seen.in_set)


class PortableWiringTests(unittest.TestCase):
    def setUp(self):
        self.sampled = import_portable_sampled()
        self.D = 16
        self.lm_f8_col = torch.empty_strided((self.D, VOCAB), (1, self.D), dtype=torch.float8_e4m3fn)
        self.lm_f8_col.T.copy_(torch.randn(VOCAB, self.D).clamp(-4, 4).to(torch.float8_e4m3fn))
        self.s = self.sampled.PortableSampledSoftmax(schedule(), VOCAB, self.D, 512, "cpu")

    def check(self, inputs, targets):
        P = inputs.rows.shape[0]
        pos = inputs.target_pos[:targets.numel()]
        self.assertTrue(((pos >= 0) & (pos < P)).all())
        self.assertTrue(torch.equal(self.s.candidates[:P][pos], targets))  # every target maps back exactly
        rows = self.lm_f8_col.T.view(torch.uint8)[self.s.candidates[:P]]
        self.assertTrue(torch.equal(inputs.rows.view(torch.uint8), rows))
        self.assertTrue(torch.equal(inputs.rows_t.view(torch.uint8), rows.T))

    def test_schedule_prunes_exactly_the_record_full_softmax_steps(self):
        self.assertEqual(sorted(self.s.pruned_steps), list(range(1107, 1195)))
        self.assertEqual(self.s.counts[1106], 24576)
        self.assertEqual(set(self.s.counts[1107:1194]), {PRUNED_WIDTH})
        self.assertIn(1107, self.s.count_change_steps())
        self.assertEqual(self.s.stage_keys(range(1107, 1194)), {(PRUNED_WIDTH, 3), (PRUNED_WIDTH, 4)})

    def test_sampled_steps_mark_and_pruned_steps_cover_every_target(self):
        rng = np.random.default_rng(0)
        early = torch.from_numpy(rng.integers(0, 40_000, 512))
        self.s.prepare(5, early, self.lm_f8_col)  # a sampled step (P = 10240) still marks its targets
        self.assertTrue(self.s.seen.seen[early].all())
        late = torch.from_numpy(rng.integers(0, 50_257, 512))
        inputs = self.s.prepare(1110, late, self.lm_f8_col, late.clone())
        self.assertEqual(inputs.rows.shape[0], PRUNED_WIDTH)
        self.check(inputs, late)
        self.assertTrue((inputs.prefix_pos == -1).all())
        builds = self.s.seen.stats["builds"]
        again = late[:100].clone()  # all already in the set: no rebuild, positions still exact
        self.check(self.s.prepare(1110, again, self.lm_f8_col), again)
        self.assertEqual(self.s.seen.stats["builds"], builds)

    def test_warmup_interleaving_restores_the_pruned_buffers(self):
        rng = np.random.default_rng(1)
        t = [torch.from_numpy(rng.integers(0, 50_257, 512)) for _ in range(4)]
        self.check(self.s.prepare(1107, t[0], self.lm_f8_col), t[0])
        self.s.prepare(1106, t[1], self.lm_f8_col)  # sampled: overwrites candidates / vocab_pos / prefix_pos
        self.check(self.s.prepare(1108, t[2], self.lm_f8_col), t[2])
        self.s.reset()
        self.assertFalse(self.s.seen.seen.any())
        self.check(self.s.prepare(1107, t[3], self.lm_f8_col), t[3])

    def test_too_many_seen_ids_fall_back_to_the_full_softmax(self):
        with patch.object(self.s.seen, "width", 1000):
            targets = torch.arange(2000)
            self.assertIsNone(self.s.prepare(1150, targets, self.lm_f8_col))
        self.assertEqual(self.s.seen.stats["full_softmax_microbatches"], 1)


class RankedModeTests(unittest.TestCase):
    def test_widths_follow_the_census_and_the_floor(self):
        self.assertEqual(ranked_width(0.001), 46_336)
        self.assertEqual(ranked_width(0.05), 19_968 + 128)
        self.assertEqual(ranked_width(0.10), 11_136)
        self.assertEqual(ranked_width(0.15), 6_656)
        self.assertEqual(ranked_width(0.20), MIN_RANKED_WIDTH)
        self.assertEqual(ranked_width(0.30), MIN_RANKED_WIDTH)
        self.assertGreater(MIN_RANKED_WIDTH, 5_292)  # census max distinct targets per microbatch
        for mass in KEPT_IDS_BY_MASS:
            self.assertEqual(ranked_width(mass) % 128, 0)
        self.assertEqual(ranked_counts([10240, 24576, 0], 0.10, VOCAB), [10240, 11136, 11136])

    def test_build_keeps_targets_and_fills_by_rank(self):
        r = RankedCandidates(20, 64, "cpu")
        r.build(0, torch.tensor([7, 7, 7, 3, 3, 9]), 6)  # step 0 counts: 7 x3, 3 x2, 9 x1
        cand = r.build(1, torch.tensor([12, 0]), 6)  # step 1 ranks by step 0's counts
        self.assertEqual(cand.tolist(), [0, 1, 3, 7, 9, 12])  # targets 0, 12; then 7, 3, 9; then id 1
        self.assertEqual(int(r.overflow), 0)
        r.build(1, torch.arange(10), 6)
        self.assertEqual(int(r.overflow), 1)
        self.assertIn("overflow_microbatches=1", r.summary())
        r.reset()
        self.assertFalse(r.counts.any())

    def test_ranked_wiring_maps_targets_and_prefixes(self):
        sampled = import_portable_sampled()
        D = 16
        lm = torch.empty_strided((D, VOCAB), (1, D), dtype=torch.float8_e4m3fn)
        lm.T.copy_(torch.randn(VOCAB, D).clamp(-4, 4).to(torch.float8_e4m3fn))
        s = sampled.PortableSampledSoftmax(schedule(), VOCAB, D, 512, "cpu", mass=0.10)
        self.assertEqual((s.counts[0], s.counts[700], s.counts[1000], s.counts[1150]), (10240, 11136, 11136, 11136))
        self.assertIn("mode=ranked mass=0.1 width=11136", s.describe())
        prefix = np.full(VOCAB, -1, dtype=np.int64)
        prefix[100:200] = np.arange(5000, 5100)  # tokens 100-199 have a prefix token
        s.set_prefix_table(prefix)
        rng = np.random.default_rng(2)
        for step in (0, 1, 700, 1150):
            targets = torch.from_numpy(rng.integers(0, 50_257, 512))
            targets[:50] = torch.arange(100, 150)
            inputs = s.prepare(step, targets, lm)
            P = s.counts[step]
            pos = inputs.target_pos[:512]
            self.assertTrue(torch.equal(s.candidates[:P][pos], targets))
            self.assertEqual(s.candidates[:P].unique().numel(), P)
            rows = lm.T.view(torch.uint8)[s.candidates[:P]]
            self.assertTrue(torch.equal(inputs.rows.view(torch.uint8), rows))
            pp = inputs.prefix_pos[:512]
            has = torch.from_numpy(prefix)[targets] >= 0
            self.assertTrue((pp[~has] == -1).all())
            kept = pp[has] >= 0
            self.assertTrue(torch.equal(s.candidates[:P][pp[has][kept]], torch.from_numpy(prefix)[targets[has]][kept]))
        self.assertIn("overflow_microbatches=0", s.summary())

    def test_config_accepts_only_census_masses(self):
        name = "track_1_short.portable.config"

        def fresh():
            sys.modules.pop(name, None)
            return importlib.import_module(name)

        original = sys.modules.get(name)
        try:
            with patch.dict(os.environ, RTX_VOCAB_PRUNE_MASS="0.05"):
                self.assertEqual(fresh().VOCAB_PRUNE_MASS, 0.05)
            with patch.dict(os.environ, RTX_VOCAB_PRUNE_MASS="0.06"), self.assertRaises(ValueError):
                fresh()
        finally:
            if original is not None:
                sys.modules[name] = original


if __name__ == "__main__":
    unittest.main()

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
    pruned_counts, ranked_counts, ranked_width, ranked_widths_in_use, weighted_kernel_source,
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
        cand, logw = r.build(1, torch.tensor([12, 0]), 6)  # step 1 ranks by step 0's counts
        self.assertIsNone(logw)
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


def record_kernel_source():
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).resolve().parents[1] / "track_1_short/perf/kernels/cross_entropy.py").read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "CE_KERNEL_SOURCE")


class StratifiedModeTests(unittest.TestCase):
    def test_weighted_kernel_is_the_record_kernel_plus_column_weights(self):
        source = record_kernel_source()
        weighted = weighted_kernel_source(source)
        self.assertEqual(weighted.count("col_logw"), 4)  # parameter, forward sum, dense grad, target grad
        self.assertIn("__expf(tmp - A + col_logw[idx + k])", weighted)
        self.assertIn("__expf(z - lse + col_logw[idx + j + t])", weighted)
        self.assertIn("__expf(z - lse + col_logw[target])", weighted)
        self.assertNotIn("col_logw", source)  # the record's source is untouched
        stripped = (weighted.replace("    const float* __restrict__ col_logw,\n", "")
                    .replace(" + col_logw[idx + k]", "").replace(" + col_logw[idx + j + t]", "")
                    .replace(" + col_logw[target]", ""))
        self.assertEqual(stripped, source)

    def test_widths_add_the_tail(self):
        self.assertEqual(ranked_counts([10240, 14336, 24576, 0], 0.15, VOCAB, 1024), [7680] * 4)
        self.assertEqual(ranked_counts([10240, 14336, 24576, 0], 0.10, VOCAB, 1024), [10240, 12160, 12160, 12160])
        self.assertEqual(ranked_widths_in_use((10240, 14336, 24576), 0.20, 1024, VOCAB), (7168,))

    def test_build_keeps_exact_part_and_draws_a_weighted_tail(self):
        V, width, tail = 2000, 300, 64
        r = RankedCandidates(V, 512, "cpu", tail)
        rng = np.random.default_rng(3)
        r.build(0, torch.from_numpy(rng.integers(0, 500, 400)), width)  # step 0 builds counts
        targets = torch.from_numpy(rng.integers(0, V, 200))
        cand, logw = r.build(1, targets, width)
        self.assertEqual(cand.numel(), width)
        self.assertEqual(cand.unique().numel(), width)
        self.assertTrue(torch.all(cand[1:] > cand[:-1]))
        self.assertTrue(torch.isin(targets, cand).all())
        in_tail = r.tail_mask[cand]
        self.assertEqual(int(in_tail.sum()), tail)
        self.assertFalse((r.exact & r.tail_mask).any())
        self.assertEqual(int(r.exact.sum()), width - tail)
        expected = float(np.log((V - (width - tail)) / tail))
        self.assertTrue(torch.allclose(logw[in_tail], torch.tensor(expected)))
        self.assertTrue((logw[~in_tail] == 0).all())

    def test_weighted_normalizer_is_unbiased(self):
        V, width, tail = 3000, 500, 128
        r = RankedCandidates(V, 512, "cpu", tail)
        torch.manual_seed(0)
        z = torch.randn(V) * 1.5
        targets = torch.randint(0, V, (150,))
        full = torch.exp(z).sum()
        estimates = []
        for draw in range(400):
            cand, logw = r.build(0, targets, width)
            estimates.append(torch.exp(z[cand] + logw).sum())
        mean = torch.stack(estimates).mean()
        self.assertLess(abs(float(mean / full) - 1), 0.02)
        # Without the weights the same sets miss most of the tail mass: the record_late / ranked bias.
        unweighted = torch.exp(z[cand]).sum()
        self.assertLess(float(unweighted / full), 0.5)

    def test_stratified_wiring_drops_tail_prefixes_and_fills_weights(self):
        sampled = import_portable_sampled()
        D = 16
        lm = torch.empty_strided((D, VOCAB), (1, D), dtype=torch.float8_e4m3fn)
        lm.T.copy_(torch.randn(VOCAB, D).clamp(-4, 4).to(torch.float8_e4m3fn))
        s = sampled.PortableSampledSoftmax(schedule(), VOCAB, D, 512, "cpu", mass=0.15, tail=1024)
        self.assertEqual(set(s.counts[:1194]), {7680})
        self.assertIn("mode=stratified mass=0.15 tail=1024 width=7680", s.describe())
        prefix = np.full(VOCAB, -1, dtype=np.int64)
        prefix[:VOCAB // 2] = np.arange(VOCAB // 2, VOCAB)  # half the ids have a prefix target
        s.set_prefix_table(prefix)
        rng = np.random.default_rng(4)
        for step in (0, 1, 700, 1150):
            targets = torch.from_numpy(rng.integers(0, 50_257, 512))
            inputs = s.prepare(step, targets, lm)
            P = s.counts[step]
            cand = s.candidates[:P]
            self.assertTrue(torch.equal(cand[inputs.target_pos[:512]], targets))
            self.assertEqual(int((inputs.col_logw > 0).sum()), 1024)
            self.assertTrue(torch.equal(inputs.col_logw > 0, s.ranked.tail_mask[cand]))
            pp = inputs.prefix_pos[:512]
            live = pp >= 0
            self.assertTrue(s.ranked.exact[cand[pp[live]]].all())  # prefix targets only on exact columns
            has = torch.from_numpy(prefix)[targets] >= 0
            dropped = has & ~s.ranked.exact[torch.from_numpy(prefix)[targets].clamp(min=0)]
            self.assertTrue((pp[dropped] == -1).all())

    def test_config_validates_the_tail(self):
        name = "track_1_short.portable.config"
        original = sys.modules.get(name)

        def fresh():
            sys.modules.pop(name, None)
            return importlib.import_module(name)

        try:
            with patch.dict(os.environ, RTX_VOCAB_PRUNE_MASS="0.15", RTX_VOCAB_PRUNE_TAIL="1024"):
                self.assertEqual(fresh().VOCAB_PRUNE_TAIL, 1024)
            for env in (dict(RTX_VOCAB_PRUNE_TAIL="1024"), dict(RTX_VOCAB_PRUNE_MASS="0.15", RTX_VOCAB_PRUNE_TAIL="1000")):
                with self.subTest(env=env), patch.dict(os.environ, env), self.assertRaises(AssertionError):
                    fresh()
        finally:
            if original is not None:
                sys.modules[name] = original


class ExactEndTests(unittest.TestCase):
    def test_exact_end_keeps_the_record_full_softmax_steps(self):
        self.assertEqual(ranked_counts([10240, 24576, 0], 0.20, VOCAB, 1024, exact_end=True), [7168, 7168, 0])
        sampled = import_portable_sampled()
        record = [sampled.candidate_count_at(schedule(), s) for s in range(1195)]
        s = sampled.PortableSampledSoftmax(schedule(), VOCAB, 16, 512, "cpu", mass=0.20, tail=1024, exact_end=True)
        full = [i for i, c in enumerate(record) if not c]
        self.assertTrue(full)
        self.assertEqual([i for i, c in enumerate(s.counts) if not c], full)
        self.assertEqual({c for c in s.counts if c}, {7168})
        self.assertIn("exact_end=1: full softmax on steps 1107-1193 (87 steps)", s.describe())
        lm = torch.empty_strided((16, VOCAB), (1, 16), dtype=torch.float8_e4m3fn)
        lm.T.copy_(torch.randn(VOCAB, 16).clamp(-4, 4).to(torch.float8_e4m3fn))
        targets = torch.randint(0, 50_257, (512,))
        self.assertIsNone(s.prepare(1150, targets, lm))
        self.assertIsNotNone(s.prepare(700, targets, lm).col_logw)

    def test_config_needs_a_mass(self):
        name = "track_1_short.portable.config"
        original = importlib.import_module(name)  # loaded under the clean env, restored below

        def fresh():
            sys.modules.pop(name, None)
            return importlib.import_module(name)

        try:
            with patch.dict(os.environ, RTX_VOCAB_PRUNE_MASS="0.20", RTX_VOCAB_PRUNE_EXACT_END="1"):
                self.assertTrue(fresh().VOCAB_PRUNE_EXACT_END)
            for env in (dict(RTX_VOCAB_PRUNE_EXACT_END="1"), dict(RTX_VOCAB_PRUNE_MASS="0.20", RTX_VOCAB_PRUNE_EXACT_END="yes")):
                with self.subTest(env=env), patch.dict(os.environ, env), self.assertRaises(AssertionError):
                    fresh()
        finally:
            sys.modules[name] = original
            sys.modules["track_1_short.portable"].config = original  # `from ... import config` reads this


if __name__ == "__main__":
    unittest.main()

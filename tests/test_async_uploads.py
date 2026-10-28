"""CPU checks of the async uploads (pinned rings, event order, uploaded values); no GPU or Triton.

The GPU run remains the real test: these check the bookkeeping, not that the copies overlap.
"""
import ast
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from track_1_short.config import LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, TRAINING_STAGES, WS_POST_YARN_EXT  # noqa: E402
from track_1_short.perf import pinned_batches  # noqa: E402
from track_1_short.portable import config as portable_config  # noqa: E402
from track_1_short.schedule import TrainingSchedule  # noqa: E402

V, D, T = 50304, 8, 2048  # the real vocabulary (the builder needs it coprime with its stride)


def load_portable_sampled():
    """portable/sampled.py with a host stand-in for the Triton transpose, its only GPU-only import."""
    transpose = types.ModuleType("track_1_short.perf.kernels.transpose")
    transpose.transpose_copy = lambda src, dst: dst.copy_(src.T)
    with mock.patch.dict(sys.modules, {"track_1_short.perf.kernels.transpose": transpose}):
        for name in ("track_1_short.portable.sampled", "track_1_short.sampled_softmax"):
            sys.modules.pop(name, None)
        return importlib.import_module("track_1_short.portable.sampled")


class FakeEvent:
    def __init__(self, log):
        self.log = log

    def synchronize(self):
        self.log.append(("sync", self))

    def record(self, stream=None):
        self.log.append(("record", self))


def same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


class HostOnly:
    """Pinned allocations become plain host tensors, events log, and every Tensor.copy_ is recorded."""

    def __init__(self):
        self.log = []
        real_empty, real_copy, log = torch.empty, torch.Tensor.copy_, self.log

        def empty(*args, pin_memory=False, **kwargs):
            return real_empty(*args, **kwargs)

        def copy_(dst, src, non_blocking=False):
            log.append(("copy", dst, src, non_blocking))
            return real_copy(dst, src, non_blocking=non_blocking)

        self.patches = [mock.patch("torch.empty", empty), mock.patch("torch.cuda.Event", lambda: FakeEvent(log)),
                        mock.patch.object(torch.Tensor, "copy_", copy_)]

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()


class PortableSampledUploadTests(unittest.TestCase):
    def setUp(self):
        self.module = load_portable_sampled()
        schedule = TrainingSchedule(TRAINING_STAGES, 1122, 20, device="cpu", cooldown_frac=LR_COOLDOWN_FRAC,
                                    split_embed_stage=SPLIT_EMBED_STAGE, ws_post_yarn_ext=WS_POST_YARN_EXT)
        self.full_softmax_step = schedule.boundaries[3][0]  # the batch taper
        self.host = HostOnly()
        self.host.__enter__()
        self.addCleanup(self.host.__exit__)
        self.sampled = self.module.PortableSampledSoftmax(schedule, V, D, T, torch.device("cpu"))
        rng = np.random.default_rng(0)
        self.prefix_table = np.where(rng.random(V) < 0.5, rng.integers(0, V, V), -1).astype(np.int64)
        self.sampled.set_prefix_table(self.prefix_table)
        self.table = torch.from_numpy(rng.integers(0, 256, (V, D), dtype=np.uint8))
        self.lm_head_f8_col = self.table.view(torch.float8_e4m3fn).T  # (D, V), strides (1, D)
        self.targets = [torch.from_numpy(rng.integers(0, V, T)) for _ in range(2 * self.module.PINNED_SLOTS + 1)]
        self.host.log.clear()

    def prepare(self, i=0, step=0):
        return self.sampled.prepare(step, self.targets[i], self.lm_head_f8_col)

    def test_uploads_are_the_built_set(self):
        inputs = self.prepare()
        P = self.sampled.counts[0]
        cand = self.sampled.candidates[:P].numpy()
        targets = self.targets[0].numpy()
        self.assertTrue(np.all(np.diff(cand) > 0) and set(targets) <= set(cand))
        np.testing.assert_array_equal(cand[inputs.target_pos[:T].numpy()], targets)
        prefix = self.prefix_table[targets]
        at = np.searchsorted(cand, prefix)
        found = (prefix >= 0) & (cand[np.minimum(at, P - 1)] == prefix)
        np.testing.assert_array_equal(inputs.prefix_pos[:T].numpy(), np.where(found, at, -1))
        np.testing.assert_array_equal(inputs.rows.view(torch.uint8).numpy(), self.table.numpy()[cand])
        np.testing.assert_array_equal(inputs.rows_t.view(torch.uint8).numpy(), self.table.numpy()[cand].T)
        expected_vocab_pos = np.full(V, -1)
        expected_vocab_pos[cand] = np.arange(P)
        np.testing.assert_array_equal(inputs.vocab_pos.numpy(), expected_vocab_pos)

    def test_async_copies_from_a_waited_slot(self):
        inputs = self.prepare()
        slot = self.sampled.slots[0]
        log = self.host.log
        sync = log.index(("sync", slot.uploaded))
        record = log.index(("record", slot.uploaded))
        staged = (slot.candidates, slot.target_pos, slot.prefix_pos)
        device = (self.sampled.candidates, inputs.target_pos, inputs.prefix_pos)
        copies = [(i, e) for i, e in enumerate(log) if e[0] == "copy"]
        slot_writes = [i for i, e in copies if any(same_storage(e[1], s) for s in staged)]
        uploads = [(i, e) for i, e in copies if any(same_storage(e[1], d) for d in device)]
        self.assertEqual(len(slot_writes), 3)
        self.assertEqual(len(uploads), 3)
        self.assertTrue(sync < min(slot_writes) and max(i for i, _ in uploads) < record)
        for (_, _, src, non_blocking), d, s in zip((e for _, e in uploads), device, staged):
            self.assertTrue(non_blocking, "a blocking H2D would drain the stream")
            self.assertTrue(same_storage(src, s))

    def test_ring_rotates_and_waits_before_reuse(self):
        n = self.module.PINNED_SLOTS
        for i in range(2 * n + 1):
            self.prepare(i)
        events = [e for e in self.host.log if e[0] in ("sync", "record")]
        order = [next(i for i, s in enumerate(self.sampled.slots) if s.uploaded is ev)
                 for kind, ev in events if kind == "sync"]
        self.assertEqual(order, [i % n for i in range(2 * n + 1)])
        for slot in self.sampled.slots:
            kinds = [kind for kind, ev in events if ev is slot.uploaded]
            self.assertEqual(kinds[::2], ["sync"] * len(kinds[::2]))
            self.assertEqual(kinds[1::2], ["record"] * len(kinds[1::2]))

    def test_full_softmax_step_uploads_nothing(self):
        self.assertEqual(self.sampled.counts[self.full_softmax_step], 0)
        self.assertIsNone(self.prepare(step=self.full_softmax_step))
        self.assertEqual(self.host.log, [])
        self.assertEqual(self.sampled.next_slot, 0)


class BatchStagingTests(unittest.TestCase):
    def test_record_default_is_unchanged(self):
        self.assertEqual(pinned_batches.PINNED_BATCH_SLOTS, 16)
        with HostOnly():
            staging = pinned_batches.PinnedBatchStaging(64, 8, torch.device("cpu"))
        self.assertEqual(len(staging.slots), 16)

    def test_ring_depth_and_reuse_order(self):
        with HostOnly() as host:
            staging = pinned_batches.PinnedBatchStaging(64, 8, torch.device("cpu"), slots=5)
            self.assertEqual(len(staging.slots), 5)
            for i in range(6):
                batch = (torch.full((64,), i, dtype=torch.int32), torch.full((64,), i, dtype=torch.int64),
                         torch.arange(8, dtype=torch.int32), torch.full((128,), i, dtype=torch.int32))
                for got, want in zip(staging.upload(*batch), batch):
                    self.assertTrue(torch.equal(got, want))
        first = staging.slots[0].uploaded
        kinds = [e[0] for e in host.log if e[0] in ("sync", "record") and e[1] is first]
        self.assertEqual(kinds, ["sync", "record", "sync", "record"])

    def test_port_ring_holds_two_steps(self):
        per_step = max(s.batch_size // portable_config.MICROBATCH_TOKENS for s in TRAINING_STAGES)
        self.assertEqual(portable_config.BATCH_STAGING_SLOTS, 2 * per_step)
        if portable_config.MICROBATCH_TOKENS == 16384:
            self.assertEqual(portable_config.BATCH_STAGING_SLOTS, 48)

    def test_trainer_uses_the_port_ring(self):
        tree = ast.parse((ROOT / "train_gpt_rtx.py").read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "PinnedBatchStaging"]
        self.assertEqual(len(calls), 1)
        slots = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}.get("slots")
        self.assertEqual(slots, "rtx.BATCH_STAGING_SLOTS")


if __name__ == "__main__":
    unittest.main()

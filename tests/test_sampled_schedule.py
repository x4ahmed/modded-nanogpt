"""CPU checks for the sampled-softmax schedule presets (RTX_SAMPLED_SCHEDULE). GPU: check_rtx5090
--sampled-schedule matched compiles and runs every candidate count during its warmup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

# sampled_softmax.py imports two GPU-only helpers at module level; stub them so it imports on CPU.
PROBE = r"""
import json, sys, types
import numpy as np
for name, attrs in {"track_1_short.perf.kernels.transpose": ["transpose_copy"],
                    "track_1_short.perf.sampled_softmax_overlap": ["CandidateStaging", "StagedCandidates"]}.items():
    module = types.ModuleType(name)
    for attr in attrs:
        setattr(module, attr, object)
    sys.modules[name] = module
from types import SimpleNamespace
import track_1_short.sampled_softmax as s

# The default 1194-step schedule's stage boundaries (1122 + 52 growth steps, then the 20-step extension).
bounds = [(0, 337), (337, 681), (681, 1107), (1107, 1174), (1174, 1194)]
schedule = SimpleNamespace(boundaries=bounds)
counts = [s.candidate_count_at(schedule, step) for step in range(1194)]

rng = np.random.default_rng(0)
# The worst microbatch measured on the first shard: 5197 distinct targets in 16384 tokens.
ids = rng.choice(50257, 5197, replace=False)
targets = rng.permutation(np.concatenate([ids, rng.choice(ids, 16384 - 5197)])).astype(np.int64)
builder = s.CandidateBuilder(50304, max(s.ALL_CANDIDATE_COUNTS), 16384)
built = {}
for p in s.ALL_CANDIDATE_COUNTS:
    candidates, target_pos, _ = builder.build(p, targets, None)
    built[p] = bool(len(candidates) == p and (candidates[target_pos] == targets).all())
print(json.dumps(dict(table={str(k): v for k, v in s.CANDIDATES_BY_STAGE.items()}, counts=counts,
                      built=built, distinct=int(np.unique(targets).size))))
"""


def probe(schedule, **extra):
    env = dict(os.environ, RTX_SAMPLED_SCHEDULE=schedule, **extra)
    return subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True)


class SampledScheduleTests(unittest.TestCase):
    def run_probe(self, schedule):
        result = probe(schedule)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_record_is_unchanged(self):
        out = self.run_probe("record")
        self.assertEqual(out["table"], {"0": [10240], "1": [10240], "2": [14336, 14336, 24576]})
        c = out["counts"]
        self.assertEqual((c[0], c[680], c[681], c[964], c[965], c[1106], c[1107], c[1193]),
                         (10240, 10240, 14336, 14336, 24576, 24576, 0, 0))

    def test_matched(self):
        out = self.run_probe("matched")
        c = out["counts"]
        self.assertEqual((c[0], c[336], c[337], c[680], c[681], c[964], c[965], c[1106], c[1107]),
                         (10240, 10240, 8192, 8192, 10240, 10240, 20480, 20480, 0))

    def test_matched_late_samples_the_taper_only(self):
        out = self.run_probe("matched_late")
        c = out["counts"]
        self.assertEqual((c[1106], c[1107], c[1173], c[1174], c[1193]), (20480, 20480, 20480, 0, 0))

    def test_record_late_changes_only_the_taper(self):
        record, late = self.run_probe("record")["counts"], self.run_probe("record_late")["counts"]
        changed = [s for s in range(1194) if record[s] != late[s]]
        self.assertEqual((changed[0], changed[-1]), (1107, 1173))  # exactly the batch-taper stage
        self.assertEqual({late[s] for s in changed}, {24576})
        self.assertEqual(late[1174], 0)  # the extension keeps the full softmax

    def test_every_count_is_legal_and_builds(self):
        for name in ("record", "record_late", "matched", "matched_late"):
            out = self.run_probe(name)
            self.assertEqual(out["distinct"], 5197)
            for p, ok in out["built"].items():
                self.assertEqual(int(p) % 2048, 0)
                self.assertTrue(ok, (name, p))

    def test_rejects_unknown_and_other_microbatch_sizes(self):
        self.assertNotEqual(probe("smaller").returncode, 0)
        self.assertNotEqual(probe("matched", RTX_MICROBATCH_TOKENS="32768").returncode, 0)
        self.assertEqual(probe("record", RTX_MICROBATCH_TOKENS="32768").returncode, 0)


if __name__ == "__main__":
    unittest.main()

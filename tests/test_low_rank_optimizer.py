"""CPU launch-dispatch checks; numerical optimizer execution is a GPU check."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


class LowRankOptimizerLaunchTests(unittest.TestCase):
    def test_actual_xtx_wrapper_keeps_tiles_and_only_changes_rank384_pipeline(self):
        path = Path(__file__).resolve().parents[1] / "track_1_short/perf/kernels/polar_express.py"
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("xtx_num_stages", "XTX")]
        launches = []

        class Kernel:
            def __getitem__(self, grid):
                def launch(**kwargs):
                    launches.append((grid, kwargs))
                return launch

        for rows in (768, 2816):
            for columns in (384, 768):
                for limit in (101376, 114688, 232448):
                    with self.subTest(rows=rows, columns=columns, limit=limit):
                        props = SimpleNamespace(shared_memory_per_block_optin=limit, shared_memory_per_block=49152)
                        mock_torch = SimpleNamespace(Tensor=torch.Tensor, cuda=SimpleNamespace(
                            get_device_properties=lambda device: props))
                        env = dict(torch=mock_torch, XTX_kernel=Kernel(), triton=SimpleNamespace(
                            cdiv=lambda x, y: (x + y - 1) // y))
                        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
                        x = torch.empty((1, rows, columns), device="meta")
                        out = torch.empty((1, columns, columns), device="meta")
                        self.assertIs(env["XTX"](x, out), out)
                        grid, config = launches[-1]
                        expected = (64, 128, 128) if columns == 384 else (128, 128, 64)
                        self.assertEqual(tuple(config[k] for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")), expected)
                        self.assertEqual(config["num_stages"], 2 if columns == 384 and limit < 114688 else 4)
                        self.assertEqual(config["num_warps"], 8)
                        self.assertEqual(config["LOWER_UPPER"], 1)
                        self.assertIs(config["A_ptr"], x)
                        self.assertEqual(grid, (((columns + expected[0] - 1) // expected[0]) *
                                                ((columns + expected[1] - 1) // expected[1]),))


if __name__ == "__main__":
    unittest.main()

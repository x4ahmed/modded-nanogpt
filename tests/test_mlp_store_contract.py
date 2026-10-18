"""CPU checks of store routing/coverage; GPU parity remains mandatory."""
import ast
from collections import Counter
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "track_1_short/perf/kernels/mlp.py"


class StoreContractTests(unittest.TestCase):
    def test_all_epilogues_use_store_helper(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "linear_relu_square_kernel")
        calls = [n for n in ast.walk(kernel) if isinstance(n, ast.Call)]
        stores = [n for n in calls if isinstance(n.func, ast.Name)
                  and n.func.id == "_store_mlp_tile"]
        self.assertEqual(len(stores), 12)
        self.assertFalse(any(isinstance(n.func, ast.Attribute) and n.func.attr == "store"
                             for n in calls))
        self.assertEqual({n.args[1].id for n in stores},
                         {"c_ptr", "aux_ptr", "aux_f8_ptr", "aux_t_ptr", "c_f8_ptr", "c_t_ptr"})

    def test_narrow_tile_dispatch_only(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        routes = [n.value for n in ast.walk(tree) if isinstance(n, ast.keyword)
                  and n.arg == "POINTER_STORES"]
        self.assertEqual(len(routes), 1)
        self.assertEqual(ast.unparse(routes[0]), "BLOCK_SIZE_N == 128")

    def test_persistent_tiles_cover_both_halves_once(self):
        # Mirror the persistent CTA loop and split epilogue, without CUDA imports.
        for m in (128, 2048, 16384):
            for n in (256, 2816):
                for bn in (128, 256):
                    nm, nn = (m + 127) // 128, (n + bn - 1) // bn
                    seen = Counter()
                    for cta in range(min(170, nm * nn)):
                        for tile in range(cta, nm * nn, 170):
                            row, col = divmod(tile, nn)
                            for half in (0, 1):
                                seen[row, 2 * col + half] += 1
                    self.assertEqual(seen, Counter({(r, c): 1 for r in range(nm)
                                                    for c in range(2 * nn)}))


if __name__ == "__main__":
    unittest.main()

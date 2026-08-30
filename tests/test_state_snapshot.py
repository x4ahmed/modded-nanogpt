from __future__ import annotations

import ast
import copy
import types
import unittest
from collections import OrderedDict
from pathlib import Path


try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
TRAIN_TREE = ast.parse(
    (ROOT / "train_gpt.py").read_text(encoding="utf-8"),
    filename="train_gpt.py",
)


def function_node(scope: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(scope):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def class_node(name: str) -> ast.ClassDef:
    for node in ast.walk(TRAIN_TREE):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def load_function(scope: ast.AST, name: str, **extra_globals):
    node = copy.deepcopy(function_node(scope, name))
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"copy": copy, "torch": torch, **extra_globals}
    exec(compile(module, "train_gpt.py", "exec"), namespace)
    return namespace[name]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class StateSnapshotTests(unittest.TestCase):
    def test_cpu_state_copy_is_independent_and_preserves_metadata(self):
        cpu_state_copy = load_function(TRAIN_TREE, "cpu_state_copy")
        source = OrderedDict(
            tensor=torch.arange(6, dtype=torch.float32).reshape(2, 3),
            nested={"tensor": torch.tensor([7.0])},
        )
        source._metadata = OrderedDict({"": {"version": 1}})

        snapshot = cpu_state_copy(source)

        self.assertIsInstance(snapshot, OrderedDict)
        self.assertEqual(snapshot._metadata, source._metadata)
        self.assertIsNot(snapshot._metadata, source._metadata)
        for original, copied in (
            (source["tensor"], snapshot["tensor"]),
            (source["nested"]["tensor"], snapshot["nested"]["tensor"]),
        ):
            self.assertEqual(copied.device.type, "cpu")
            self.assertTrue(torch.equal(copied, original))
            self.assertNotEqual(copied.data_ptr(), original.data_ptr())

        source["tensor"].zero_()
        self.assertTrue(torch.equal(snapshot["tensor"], torch.arange(6).reshape(2, 3)))

    def test_optimizer_restore_is_in_place_and_exact(self):
        load_state_dict = load_function(
            class_node("NorMuonAndAdam"),
            "load_state_dict",
        )
        param = torch.nn.Parameter(torch.zeros(2, 3))
        target = torch.zeros(2, 3, dtype=torch.float32)
        optimizer = types.SimpleNamespace(
            param_cfgs={param: object()},
            param_states={param: {"step": 0, "exp_avg": target, "metadata": []}},
        )
        saved_metadata = ["initial"]
        state = {
            "param_states": {
                id(param): {
                    "step": 3,
                    "exp_avg": torch.full((2, 3), 4.0, dtype=torch.float64),
                    "metadata": saved_metadata,
                }
            }
        }
        original_identity = id(target)
        original_storage = target.data_ptr()

        load_state_dict(optimizer, state)

        restored = optimizer.param_states[param]
        self.assertEqual(id(restored["exp_avg"]), original_identity)
        self.assertEqual(restored["exp_avg"].data_ptr(), original_storage)
        self.assertEqual(restored["exp_avg"].dtype, torch.float32)
        self.assertTrue(torch.equal(restored["exp_avg"], torch.full((2, 3), 4.0)))
        self.assertEqual(restored["step"], 3)
        self.assertEqual(restored["metadata"], saved_metadata)
        self.assertIsNot(restored["metadata"], saved_metadata)

    def test_optimizer_restore_rejects_shape_mismatch(self):
        load_state_dict = load_function(
            class_node("NorMuonAndAdam"),
            "load_state_dict",
        )
        param = torch.nn.Parameter(torch.zeros(2, 3))
        optimizer = types.SimpleNamespace(
            param_cfgs={param: object()},
            param_states={param: {"exp_avg": torch.zeros(2, 3)}},
        )
        state = {
            "param_states": {
                id(param): {"exp_avg": torch.zeros(3, 2)},
            }
        }

        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            load_state_dict(optimizer, state)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_cpu_optimizer_snapshot_creates_no_cuda_tensor_copy(self):
        cpu_state_copy = load_function(TRAIN_TREE, "cpu_state_copy")
        get_state = load_function(
            class_node("TrainingManager"),
            "get_state",
            cpu_state_copy=cpu_state_copy,
        )
        source = torch.arange(1 << 20, dtype=torch.float32, device="cuda")
        manager = types.SimpleNamespace(
            optimizer=types.SimpleNamespace(state_dict=lambda: {"tensor": source})
        )
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        snapshot = get_state(manager, to_cpu=True)
        torch.cuda.synchronize()

        self.assertEqual(snapshot["tensor"].device.type, "cpu")
        self.assertTrue(torch.equal(snapshot["tensor"], source.cpu()))
        self.assertEqual(torch.cuda.max_memory_allocated(), allocated_before)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_cpu_optimizer_restore_reuses_cuda_storage(self):
        load_state_dict = load_function(
            class_node("NorMuonAndAdam"),
            "load_state_dict",
        )
        param = torch.nn.Parameter(torch.zeros(1))
        target = torch.zeros(1 << 20, dtype=torch.float32, device="cuda")
        optimizer = types.SimpleNamespace(
            param_cfgs={param: object()},
            param_states={param: {"exp_avg": target}},
        )
        state = {
            "param_states": {
                id(param): {"exp_avg": torch.full((1 << 20,), 5.0)},
            }
        }
        original_storage = target.data_ptr()
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        load_state_dict(optimizer, state)
        torch.cuda.synchronize()

        restored = optimizer.param_states[param]["exp_avg"]
        self.assertEqual(restored.data_ptr(), original_storage)
        self.assertTrue(torch.equal(restored.cpu(), state["param_states"][id(param)]["exp_avg"]))
        self.assertEqual(torch.cuda.max_memory_allocated(), allocated_before)


if __name__ == "__main__":
    unittest.main()

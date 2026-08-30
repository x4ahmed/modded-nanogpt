"""Structural invariants of the RTX-local-v1 port, checked against the parsed AST.

These assert *structure*, not source text: importing train_gpt.py needs CUDA and an
initialized process group, so the pinned schedule, the PORTABLE guards and the CE
shared-memory arithmetic are read out of the syntax tree instead. Substring matching
was deliberately avoided here — it passes on semantically broken code and fails on a
reindent. Numeric behavior lives in test_portable_attention.py and the executable
preflight; this file only guarantees the invariants those tests assume.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse(name: str) -> ast.Module:
    return ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)


TRAIN_TREE = parse("train_gpt.py")
ATTENTION_TREE = parse("portable_attention.py")
TRITON_TREE = parse("triton_kernels.py")
PORTABLE_PREFLIGHT_TREE = parse("portable_preflight.py")
PREFLIGHT_TREE = parse("tests/rtx_preflight.py")
PREFLIGHT_SRC = (ROOT / "tests" / "rtx_preflight.py").read_text(encoding="utf-8")

# RTX 5090 / SM120 per-block dynamic shared-memory opt-in ceiling, in bytes.
SM120_SHARED_MEMORY_OPTIN_LIMIT = 101_376


def class_def(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def function_def(scope: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(scope):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def annotated_default(cls: ast.ClassDef, field: str) -> ast.expr:
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == field:
            assert node.value is not None, f"{field} has no default"
            return node.value
    raise AssertionError(f"field {field} not found on {cls.name}")


def module_assignment(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == name for target in node.targets
        ):
            return node.value
    raise AssertionError(f"module-level assignment {name} not found")


def first_lineno(scope: ast.AST, *, call: str | None = None, attribute: str | None = None) -> int:
    linenos = []
    for node in ast.walk(scope):
        if call is not None and isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == call:
                linenos.append(node.lineno)
        if attribute is not None and isinstance(node, ast.Attribute) and node.attr == attribute:
            linenos.append(node.lineno)
    assert linenos, f"no reference to {call or attribute} in scope"
    return min(linenos)


def guarded_by(tree: ast.Module, *, assigns: str, test_src: str) -> bool:
    """True when `assigns = ...` appears only under an `if <test_src>:` branch."""
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == assigns for target in node.targets)
    ]
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or ast.unparse(node.test) != test_src:
            continue
        for inner in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if isinstance(inner, ast.Assign) and any(
                getattr(target, "id", None) == assigns for target in inner.targets
            ):
                guarded.add(id(inner))
    return bool(assignments) and all(id(node) in guarded for node in assignments)


class ScheduleContractTests(unittest.TestCase):
    def setUp(self):
        self.hyperparameters = class_def(TRAIN_TREE, "Hyperparameters")

    def test_pinned_schedule_lengths_are_unchanged(self):
        self.assertEqual(
            ast.literal_eval(annotated_default(self.hyperparameters, "num_scheduled_iterations")), 1270
        )
        self.assertEqual(
            ast.literal_eval(annotated_default(self.hyperparameters, "num_extension_iterations")), 15
        )

    def test_training_schedule_is_built_from_the_full_pinned_length(self):
        call = module_assignment(TRAIN_TREE, "training_schedule")
        self.assertIsInstance(call, ast.Call)
        rendered = ast.unparse(call)
        # A truncated run must cap the loop, never re-derive the schedule.
        self.assertNotIn("max_steps", rendered)
        self.assertNotIn("run_plan", rendered)
        self.assertIn("args.num_scheduled_iterations", rendered)
        self.assertIn("args.num_extension_iterations", rendered)

    def test_short_runs_truncate_and_skip_the_final_window_extension(self):
        loop = next(
            node
            for node in ast.walk(TRAIN_TREE)
            if isinstance(node, ast.For) and "run_plan.iterations()" in ast.unparse(node.iter)
        )
        guards = [
            ast.unparse(node.test)
            for node in ast.walk(loop)
            if isinstance(node, ast.If) and "apply_final_ws_ext" in ast.unparse(node)
        ]
        self.assertTrue(
            any("run_plan.complete_schedule" in guard for guard in guards),
            "the 13->20 window extension must be gated on the complete schedule",
        )

    def test_portable_validation_changes_packing_not_token_count(self):
        self.assertEqual(ast.literal_eval(annotated_default(self.hyperparameters, "val_tokens")), 10485760)
        self.assertEqual(
            ast.unparse(annotated_default(self.hyperparameters, "val_batch_size")),
            "validation_batch_size(PORTABLE)",
        )


class PortableGuardTests(unittest.TestCase):
    def test_native_flash_attention_import_is_portable_guarded(self):
        self.assertTrue(
            guarded_by(TRAIN_TREE, assigns="flash_attn_interface", test_src="not PORTABLE"),
            "the module-scope FA3 load must not run under PORTABLE=1",
        )

    def test_layer_window_and_block_mask_mapping_is_pinned(self):
        forward = function_def(class_def(TRAIN_TREE, "GPT"), "forward")
        portable_branch = next(
            node
            for node in ast.walk(forward)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "PORTABLE"
            and "create_document_block_masks" in ast.unparse(node)
        )
        portable_body = ast.Module(body=portable_branch.body, type_ignores=[])
        assignments = {
            ast.unparse(target): node.value
            for node in ast.walk(forward)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertEqual(
            [ast.unparse(item) for item in assignments["bm_sizes"].elts],
            ["ws_short", "ws_short", "ws_short", "ws_long", "ws_short", "ws_short", "None",
             "ws_short", "ws_short", "ws_short", "ws_long"],
        )
        portable_mapping = next(
            node.value
            for node in ast.walk(portable_body)
            if isinstance(node, ast.Assign)
            and any(getattr(target, "id", None) == "block_masks" for target in node.targets)
            and isinstance(node.value, ast.List)
            and len(node.value.elts) == 11
        )
        self.assertEqual(
            [ast.unparse(item) for item in portable_mapping.elts],
            ["paired_short_mask", "normal_short_mask", "paired_short_mask", "normal_long_mask",
             "normal_short_mask", "paired_short_mask", "None", "normal_short_mask",
             "normal_short_mask", "paired_short_mask", "normal_long_mask"],
        )

        mask_creators = [
            node
            for node in ast.walk(portable_body)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "create_document_block_masks"
        ]
        self.assertEqual(len(mask_creators), 2)
        creators_by_target = {ast.unparse(node.targets[0]): node.value for node in mask_creators}
        normal_creator = creators_by_target["(normal_short_mask, normal_long_mask)"]
        paired_creator = creators_by_target["(paired_short_mask,)"]
        self.assertEqual(ast.unparse(normal_creator.args[2]), "(ws_short, ws_long)")
        self.assertEqual(ast.unparse(paired_creator.args[2]), "(ws_short,)")
        self.assertEqual(ast.unparse(normal_creator.keywords[0].value), "False")
        self.assertEqual(ast.unparse(paired_creator.keywords[0].value), "True")

    def test_flex_attention_pins_kernel_options(self):
        forward = function_def(class_def(TRAIN_TREE, "CausalSelfAttention"), "forward")
        call = next(
            node
            for node in ast.walk(forward)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "flex_attention"
        )
        # Inductor's default tiles exceed SM120's shared-memory limit; the blocks
        # must be pinned explicitly or compilation fails with no valid triton configs.
        self.assertIn("kernel_options", [kw.arg for kw in call.keywords])

    def test_paired_head_layers_are_pinned(self):
        block = next(
            node
            for node in ast.walk(TRAIN_TREE)
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "self.paired_head_layers"
        )
        self.assertEqual(ast.literal_eval(block.value), [0, 2, 5, 9])

    def test_post_attention_order_is_preserved(self):
        forward = function_def(class_def(TRAIN_TREE, "CausalSelfAttention"), "forward")
        flex = first_lineno(forward, call="flex_attention")
        dc = first_lineno(forward, call="dc_attention_postonly_nodd_correction_add_base_triton")
        xsa = first_lineno(forward, attribute="xsa_alpha")
        gate = next(
            node.lineno
            for node in ast.walk(forward)
            if isinstance(node, ast.If) and "attn_gate_w is not None" in ast.unparse(node.test)
        )
        output_projection = max(
            node.lineno
            for node in ast.walk(forward)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "F.linear"
        )
        self.assertLess(flex, dc, "the DC correction must follow attention")
        self.assertLess(dc, xsa, "gated XSA must follow the DC correction")
        self.assertLess(xsa, gate, "the raw attention gate must follow XSA")
        self.assertLess(gate, output_projection, "the output projection must remain last")
        gpt_forward = function_def(class_def(TRAIN_TREE, "GPT"), "forward")
        self.assertIn("cache.get(7, x)", ast.unparse(gpt_forward))

    def test_optimizer_groups_remain_structurally_pinned(self):
        init = function_def(class_def(TRAIN_TREE, "TrainingManager"), "__init__")
        assignment = next(
            node
            for node in ast.walk(init)
            if isinstance(node, ast.Assign)
            and any(ast.unparse(target) == "self.param_table" for target in node.targets)
        )
        groups = ast.literal_eval(assignment.value)
        for node in ast.walk(init):
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func) == "self.param_table.update"
                and node.args
                and isinstance(node.args[0], ast.Dict)
            ):
                groups.update(ast.literal_eval(node.args[0]))
        self.assertEqual(
            set(groups),
            {"qk_bank", "vo_bank", "mlp_bank", "scalars", "smear_gate", "ve_gate_bank",
             "lm_head", "bigram_embed", "post_lambdas", "resid_lambdas", "value_embeds", "embed",
             "mudd_w1", "mudd_w2", "mudd_b2", "mudd_gate_w1", "mudd_gate_w2",
             "mudd_gate_b2", "_mudd_gate_scale"},
        )
        self.assertEqual(groups["qk_bank"]["optim"], "normuon")
        self.assertEqual(groups["mlp_bank"]["comms"], "sharded")
        self.assertEqual(groups["scalars"]["optim"], "adam")
        self.assertEqual(groups["bigram_embed"]["comms"], "sharded_sparse")
        self.assertEqual(groups["mudd_w1"]["lr_mul"], 0.25)


class AttentionMaskContractTests(unittest.TestCase):
    def test_window_predicate_is_inclusive_and_never_doubled(self):
        mask_mod = function_def(ATTENTION_TREE, "document_causal_window")
        rendered = ast.unparse(mask_mod)
        # FA3 is called with window_size=(bm_size, 0), i.e. q - k <= bm_size inclusive.
        self.assertIn("q_idx - kv_idx <= window_tokens", rendered)
        self.assertIn("q_idx >= kv_idx", rendered)
        self.assertIn("document_ids[q_idx] == document_ids[kv_idx]", rendered)
        # Paired heads double positions, not window widths.
        self.assertNotIn("window_tokens * 2", ast.unparse(ATTENTION_TREE))

    def test_paired_masks_double_positions(self):
        creator = function_def(ATTENTION_TREE, "create_document_block_masks")
        self.assertIn("repeat_interleave(2)", ast.unparse(creator))

    def test_block_size_matches_the_schedule_block_unit(self):
        self.assertEqual(ast.literal_eval(module_assignment(ATTENTION_TREE, "BLOCK_SIZE")), 128)


class CeSharedMemoryContractTests(unittest.TestCase):
    def constant(self, name: str) -> int:
        return self._resolve(module_assignment(TRITON_TREE, name))

    def _resolve(self, node: ast.expr) -> int:
        """Fold the pinned integer arithmetic without importing the CUDA module."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.constant(node.id)
        if isinstance(node, ast.BinOp):
            left, right = self._resolve(node.left), self._resolve(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Mult):
                return left * right
        raise AssertionError(f"unsupported constant expression: {ast.unparse(node)}")

    def test_shared_memory_budget_fits_sm120(self):
        vocab = self.constant("CE_KERNEL_VOCAB_SIZE")
        dynamic = self.constant("CE_KERNEL_DYNAMIC_SHARED_BYTES")
        static = self.constant("CE_KERNEL_STATIC_SHARED_BYTES_MIN")
        self.assertEqual(vocab, 50304)
        self.assertEqual(dynamic, 100_608)
        # Two __shared__ float[NUM_WARPS] arrays, NUM_WARPS = BLOCK_SIZE(256) / 32 = 8.
        self.assertEqual(static, 64)
        total = dynamic + static
        self.assertEqual(self.constant("CE_KERNEL_TOTAL_SHARED_BYTES_MIN"), total)
        self.assertLessEqual(
            total,
            SM120_SHARED_MEMORY_OPTIN_LIMIT,
            "CE kernel cannot fit SM120's per-block opt-in limit",
        )

    def test_fused_mlp_tiles_are_chosen_from_the_measured_device_limit(self):
        chooser = function_def(TRITON_TREE, "_mlp_block_config")
        rendered = ast.unparse(chooser)
        # Selection must come from the device's own limit compared against the
        # configuration's computed requirement -- never a hardcoded cutoff, a device
        # name, or PORTABLE. A fixed threshold misclassifies the A100.
        self.assertIn("select_mlp_block_config(_device_shared_memory_limit(device)", rendered)
        self.assertNotIn("PORTABLE", rendered)
        self.assertNotIn("_H100_CLASS_SHARED_MEMORY", ast.unparse(TRITON_TREE))
        for banned in ("5090", "sm_120", "H100", "A100"):
            self.assertNotIn(banned, rendered)
        # This runs inside the compiled region, so memoizing here is a dynamo graph
        # break ("Mutating a variable not in the current scope"). Keep it pure.
        subscript_assignments = [
            ast.unparse(target)
            for node in ast.walk(chooser)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Subscript)
        ]
        self.assertEqual(
            subscript_assignments, [], "no memoization inside the traced path"
        )

    def test_launch_caps_pipeline_depth_to_the_chosen_config(self):
        launcher = function_def(TRITON_TREE, "linear_relu_square")
        rendered = ast.unparse(launcher)
        # num_stages must never exceed what the device config allows.
        self.assertIn("min(4 if FORWARD else 3, MAX_STAGES)", rendered)
        self.assertIn("min(3, MAX_STAGES)", rendered)
        self.assertNotIn("num_stages = 4 if FORWARD else 3", rendered)

    def test_compute_target_is_unchanged_pending_device_evidence(self):
        self.assertEqual(
            ast.literal_eval(module_assignment(TRITON_TREE, "CE_KERNEL_COMPUTE_CAPABILITY")), "90"
        )

    def test_shared_memory_optin_precedes_launch_without_hot_path_reconfiguration(self):
        compile_kernel = function_def(TRITON_TREE, "compile_ce_kernel")
        cache_guard = next(
            node
            for node in ast.walk(compile_kernel)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "_ce_fwd_bwd_kernel is None"
        )
        compile_calls = [
            node
            for node in ast.walk(compile_kernel)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "_compile_kernel"
        ]
        self.assertEqual(len(compile_calls), 1)
        self.assertLessEqual(cache_guard.lineno, compile_calls[0].lineno)
        self.assertLessEqual(compile_calls[0].lineno, cache_guard.end_lineno)

        configure = function_def(TRITON_TREE, "configure_ce_kernel")
        configure_rendered = ast.unparse(configure)
        self.assertLess(
            configure_rendered.index("compile_ce_kernel()"),
            configure_rendered.index("set_shared_memory_config(CE_KERNEL_DYNAMIC_SHARED_BYTES)"),
        )
        setters = [
            node
            for node in ast.walk(TRITON_TREE)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "set_shared_memory_config"
        ]
        self.assertEqual(len(setters), 1)
        self.assertLessEqual(configure.lineno, setters[0].lineno)
        self.assertLessEqual(setters[0].lineno, configure.end_lineno)

        eager_init = next(
            node
            for node in TRITON_TREE.body
            if isinstance(node, ast.If) and "_RTX_DEFER_CE_INIT" in ast.unparse(node.test)
        )
        self.assertEqual(ast.unparse(eager_init.test), "os.environ.get('_RTX_DEFER_CE_INIT') != '1'")
        self.assertIn("configure_ce_kernel()", ast.unparse(eager_init))

        launcher = function_def(TRITON_TREE, "ce_fwd_bwd")
        rendered = ast.unparse(launcher)
        self.assertNotIn("configure_ce_kernel()", rendered)
        self.assertNotIn("set_shared_memory_config", rendered)
        self.assertIn("shared_mem=CE_KERNEL_DYNAMIC_SHARED_BYTES", rendered)

        preflight = function_def(PORTABLE_PREFLIGHT_TREE, "run_ce_preflight")
        stages = {
            node.args[0].value: node
            for node in ast.walk(preflight)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "stage"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        }
        self.assertEqual(ast.unparse(stages["CE_COMPILE"].args[1]), "compile_ce_kernel")
        self.assertEqual(ast.unparse(stages["CE_SHARED_CONFIG"].args[1]), "configure_ce_kernel")
        self.assertIn("ce_fwd_bwd", ast.unparse(stages["CE_LAUNCH"].args[1]))
        self.assertLess(stages["CE_COMPILE"].lineno, stages["CE_SHARED_CONFIG"].lineno)
        self.assertLess(stages["CE_SHARED_CONFIG"].lineno, stages["CE_LAUNCH"].lineno)


class PreflightContractTests(unittest.TestCase):
    def test_preflight_execs_training_without_importing_it(self):
        self.assertIn("os.execve", ast.unparse(PREFLIGHT_TREE))
        self.assertNotRegex(PREFLIGHT_SRC, re.compile(r"^\s*(?:from|import)\s+train_gpt", re.MULTILINE))

    def test_preflight_refuses_fallbacks_and_wrong_world_size(self):
        rendered = ast.unparse(PREFLIGHT_TREE)
        # 1 and 2 ranks are both gateable; anything else must be refused.
        self.assertIn("os.environ.get('WORLD_SIZE')", rendered)
        self.assertIn("{'1', '2'}", rendered)
        self.assertIn("os.environ.get('LOCAL_WORLD_SIZE') != world_size", rendered)
        self.assertIn("os.environ.get('DISABLE_FP8')", rendered)

    def test_memory_gate_spans_warmup_update_and_validation(self):
        resets = [
            node.lineno
            for node in ast.walk(TRAIN_TREE)
            if isinstance(node, ast.Call) and ast.unparse(node) == "torch.cuda.reset_peak_memory_stats(device)"
        ]
        self.assertEqual(len(resets), 1, "one reset only, or the gate measures a partial peak")
        warmup = next(
            node.lineno
            for node in ast.walk(TRAIN_TREE)
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "initial_state" for t in node.targets)
        )
        gate = next(
            node
            for node in ast.walk(TRAIN_TREE)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "RTX_PREFLIGHT"
            and "MEMORY_COMPILE_WARMUP_UPDATE_VALIDATION" in ast.unparse(node)
        )
        self.assertLess(resets[0], warmup, "peak stats must reset before compile and warmup")
        self.assertLess(warmup, gate.lineno, "the gate must cover compile and warmup")

        gate_body = ast.Module(body=gate.body, type_ignores=[])
        rendered = ast.unparse(gate_body)
        self.assertIn("model.eval()", rendered)
        self.assertIn("apply_final_ws_ext()", rendered)
        self.assertIn("args.val_batch_size", rendered)
        self.assertIn("odd_update_step = training_schedule.scheduled_iterations + 1", rendered)
        self.assertIn("terminal_even_step = training_schedule.total_steps - 1", rendered)
        self.assertIn("odd_update_step % 2 == 1", rendered)
        self.assertIn("terminal_even_step % 2 == 0", rendered)
        self.assertIn("preflight_update_steps = (odd_update_step, terminal_even_step)", rendered)
        self.assertIn("quantize_mlp_fp8(bootstrap_down=preflight_step < 16)", rendered)
        self.assertIn("advance_schedule(training_schedule.total_steps)", rendered)

        update_loop = next(
            node
            for node in ast.walk(gate_body)
            if isinstance(node, ast.For) and ast.unparse(node.iter) == "preflight_update_steps"
        )
        update_loop_rendered = ast.unparse(update_loop)
        self.assertIn("training_manager.step_optimizers(preflight_step)", update_loop_rendered)
        self.assertIn(
            "model.quantize_mlp_fp8(bootstrap_down=preflight_step < 16)",
            update_loop_rendered,
        )
        zero_grad_lines = [
            node.lineno
            for node in ast.walk(gate_body)
            if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("zero_grad")
        ]
        final_window = first_lineno(gate_body, call="apply_final_ws_ext")
        validation_loop = next(
            node
            for node in ast.walk(gate_body)
            if isinstance(node, ast.For)
            and node.lineno > final_window
            and "next(val_loader)" in ast.unparse(node)
            and "model(inputs, targets" in ast.unparse(node)
        )
        post_validation_sync = min(
            node.lineno
            for node in ast.walk(gate_body)
            if isinstance(node, ast.Call)
            and ast.unparse(node) == "torch.cuda.synchronize()"
            and node.lineno > validation_loop.end_lineno
        )
        peak_read = max(
            node.lineno
            for node in ast.walk(gate_body)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "max_memory_reserved"
        )
        self.assertEqual(len(zero_grad_lines), 1)
        self.assertLess(zero_grad_lines[0], update_loop.lineno)
        self.assertLess(update_loop.end_lineno, final_window)
        self.assertLess(final_window, validation_loop.lineno)
        self.assertLess(validation_loop.end_lineno, post_validation_sync)
        self.assertLess(post_validation_sync, peak_read)
        self.assertLess(final_window, peak_read)
        self.assertIn("odd_update_step=", rendered)
        self.assertIn("terminal_even_step=", rendered)
        self.assertIn("validation_step=", rendered)

        optimizer_step = function_def(class_def(TRAIN_TREE, "NorMuonAndAdam"), "step")
        retain_adam_grads = next(
            node
            for node in ast.walk(optimizer_step)
            if isinstance(node, ast.If)
            and "p_cfg.optim == 'adam' and (not do_adam)" in ast.unparse(node.test)
            and any(isinstance(inner, ast.Continue) for inner in ast.walk(node))
        )
        clear_grad = next(
            node
            for node in ast.walk(optimizer_step)
            if isinstance(node, ast.Assign)
            and any(ast.unparse(target) == "param.grad" for target in node.targets)
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
        )
        self.assertLess(retain_adam_grads.lineno, clear_grad.lineno)


class LauncherContractTests(unittest.TestCase):
    def test_run_sh_forwards_cli_arguments(self):
        self.assertIn('train_gpt.py "$@"', (ROOT / "run.sh").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

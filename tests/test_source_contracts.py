from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = (ROOT / "train_gpt.py").read_text(encoding="utf-8")
ATTENTION = (ROOT / "portable_attention.py").read_text(encoding="utf-8")
TRITON = (ROOT / "triton_kernels.py").read_text(encoding="utf-8")
PREFLIGHT = (ROOT / "tests" / "rtx_preflight.py").read_text(encoding="utf-8")


class SourceContractTests(unittest.TestCase):
    def test_short_run_caps_only_the_loop(self):
        self.assertIn("num_scheduled_iterations: int = 1270", TRAIN)
        self.assertIn("num_extension_iterations: int = 15", TRAIN)
        schedule_line = next(
            line for line in TRAIN.splitlines() if line.startswith("training_schedule = TrainingSchedule")
        )
        self.assertNotIn("max_steps", schedule_line)
        self.assertIn("for step, last_step in run_plan.iterations():", TRAIN)
        self.assertIn("if last_step and run_plan.complete_schedule:", TRAIN)
        self.assertIn("ws_post_yarn_ext: int = 20", TRAIN)

    def test_portable_validation_changes_packing_not_token_count(self):
        self.assertIn("val_tokens: int = 10485760", TRAIN)
        self.assertIn("val_batch_size: int = validation_batch_size(PORTABLE)", TRAIN)

    def test_native_flash_import_is_guarded(self):
        self.assertIn("if not PORTABLE:\n    flash_attn_interface = get_kernel", TRAIN)
        self.assertIn("if PORTABLE:\n            assert attn_args.block_mask is not None", TRAIN)

    def test_attention_mask_is_exact_and_paired_window_is_not_doubled(self):
        self.assertIn("q_idx - kv_idx <= window_tokens", ATTENTION)
        self.assertIn("window_any = q0 - k1 <= window_tokens", ATTENTION)
        self.assertIn("window_all = q1 - k0 <= window_tokens", ATTENTION)
        self.assertIn("boundaries = seqlens[1:].to(torch.int64)", ATTENTION)
        self.assertIn("document_starts.scatter_add_", ATTENTION)
        self.assertIn("document_ids = document_ids.repeat_interleave(2)", ATTENTION)
        self.assertNotIn("window_tokens * 2", ATTENTION)

    def test_transformer_topology_and_post_attention_order_are_preserved(self):
        self.assertIn("self.paired_head_layers = [0, 2, 5, 9]", TRAIN)
        self.assertIn("attn_in_normed = norm(cache.get(7, x))", TRAIN)
        attention_start = TRAIN.index("class CausalSelfAttention")
        attention_end = TRAIN.index("# The main model", attention_start)
        attention_forward = TRAIN[attention_start:attention_end]
        self.assertLess(attention_forward.index("flex_attention("), attention_forward.index("dc_attention_postonly"))
        self.assertLess(attention_forward.index("dc_attention_postonly"), attention_forward.index("# Gated XSA"))
        self.assertLess(attention_forward.index("# Gated XSA"), attention_forward.index("attn_gate_w.type_as"))

    def test_optimizer_groups_remain_structurally_pinned(self):
        for group in (
            '"qk_bank":',
            '"vo_bank":',
            '"mlp_bank":',
            '"scalars":',
            '"lm_head":',
            '"bigram_embed":',
            '"mudd_w1":',
        ):
            self.assertIn(group, TRAIN)

    def test_ce_shared_memory_contract_and_target(self):
        self.assertRegex(TRITON, r"CE_KERNEL_VOCAB_SIZE\s*=\s*50304")
        self.assertIn('CE_KERNEL_COMPUTE_CAPABILITY = "90"', TRITON)
        self.assertIn("CE_KERNEL_DYNAMIC_SHARED_BYTES = CE_KERNEL_VOCAB_SIZE * 2", TRITON)
        self.assertIn("CE_KERNEL_STATIC_SHARED_BYTES_MIN = 2 * 8 * 4", TRITON)
        self.assertIn("kernel.set_shared_memory_config(CE_KERNEL_DYNAMIC_SHARED_BYTES)", TRITON)
        self.assertIn('if os.environ.get("_RTX_DEFER_CE_INIT") != "1":', TRITON)
        self.assertIn("    configure_ce_kernel()", TRITON)

    def test_preflight_execs_training_without_importing_it(self):
        self.assertIn("os.execve(sys.executable, child_argv, child_env)", PREFLIGHT)
        self.assertNotRegex(PREFLIGHT, re.compile(r"^\s*(?:from|import)\s+train_gpt", re.MULTILINE))
        self.assertIn('os.environ.get("WORLD_SIZE") != "2"', PREFLIGHT)
        self.assertIn('if os.environ.get("DISABLE_FP8"):', PREFLIGHT)
        self.assertIn("PYTHONHASHSEED=str(args.seed)", PREFLIGHT)

    def test_memory_gate_includes_compile_warmup_and_update(self):
        reset_index = TRAIN.index("torch.cuda.reset_peak_memory_stats(device)")
        warmup_state_index = TRAIN.index("initial_state = dict(")
        measured_update_index = TRAIN.index('"MEMORY_COMPILE_WARMUP_UPDATE "')
        self.assertLess(reset_index, warmup_state_index)
        self.assertLess(warmup_state_index, measured_update_index)
        self.assertEqual(TRAIN.count("torch.cuda.reset_peak_memory_stats(device)"), 1)

    def test_run_sh_forwards_cli_arguments(self):
        run_sh = (ROOT / "run.sh").read_text(encoding="utf-8")
        self.assertIn('train_gpt.py "$@"', run_sh)


if __name__ == "__main__":
    unittest.main()

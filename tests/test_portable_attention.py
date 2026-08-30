from __future__ import annotations

import sys
import unittest
from pathlib import Path

if __package__:
    from .gpu import require_triton, requires_cuda
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gpu import require_triton, requires_cuda

try:
    import torch
except ImportError:  # CPU-only development environments may not install project deps.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BlockMaskSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from portable_attention import _build_document_block_mask

        cls.build_mask = staticmethod(_build_document_block_mask)

    def test_inclusive_window_and_document_boundary(self):
        document_ids = torch.ones(256, dtype=torch.int64)
        document_ids[160:] = 2
        mask = self.build_mask(document_ids, 128)
        zero = torch.tensor(0)

        def allowed(q, kv):
            return bool(mask.mask_mod(zero, zero, torch.tensor(q), torch.tensor(kv)))

        self.assertTrue(allowed(159, 31))   # inclusive q-kv == W
        self.assertFalse(allowed(159, 30))  # q-kv == W+1
        self.assertFalse(allowed(200, 159)) # previous document
        self.assertFalse(allowed(100, 101)) # future token
        self.assertTrue(allowed(200, 160))

    def test_paired_documents_double_positions_but_not_window(self):
        from portable_attention import create_document_block_masks

        seqlens = torch.tensor([0, 160, 256, 256], dtype=torch.int32)
        (mask,) = create_document_block_masks(seqlens, 256, (128,), paired=True)
        zero = torch.tensor(0)
        allowed = lambda q, kv: bool(
            mask.mask_mod(zero, zero, torch.tensor(q), torch.tensor(kv))
        )
        self.assertTrue(allowed(300, 172))
        self.assertFalse(allowed(300, 171))
        self.assertFalse(allowed(400, 319))
        self.assertTrue(allowed(400, 320))

    def test_end_exclusive_seqlens_control_delimiter_ownership(self):
        from portable_attention import create_document_block_masks

        # A boundary at 160 means token 159 (for example EOT) remains in the
        # preceding document and token 160 starts the next one.
        seqlens = torch.tensor([0, 160, 256, 256], dtype=torch.int32)
        (mask,) = create_document_block_masks(seqlens, 256, (256,), paired=False)
        zero = torch.tensor(0)
        allowed = lambda q, kv: bool(
            mask.mask_mod(zero, zero, torch.tensor(q), torch.tensor(kv))
        )
        self.assertTrue(allowed(159, 0))
        self.assertFalse(allowed(160, 159))
        self.assertTrue(allowed(160, 160))

    def test_partial_and_full_block_tables_match_exhaustive_oracle(self):
        block_size, length = 128, 512
        document_ids = torch.zeros(length, dtype=torch.int64)
        for boundary in (0, 37, 160, 289, 420):
            document_ids[boundary:] += 1
        positions = torch.arange(length)

        def ordered_to_dense(num_blocks, indices):
            dense = torch.zeros(indices.shape[-2:], dtype=torch.bool)
            counts = num_blocks[0, 0]
            ordered = indices[0, 0]
            for row in range(ordered.shape[0]):
                dense[row, ordered[row, : int(counts[row])]] = True
            return dense

        for window in (128, 256, 384):
            with self.subTest(window=window):
                mask = self.build_mask(document_ids, window)
                token_oracle = (
                    (positions[:, None] >= positions)
                    & (positions[:, None] - positions <= window)
                    & (document_ids[:, None] == document_ids)
                )
                tiled = token_oracle.view(
                    length // block_size,
                    block_size,
                    length // block_size,
                    block_size,
                ).permute(0, 2, 1, 3)
                any_oracle = tiled.any(dim=-1).any(dim=-1)
                full_oracle = tiled.all(dim=-1).all(dim=-1)
                partial_oracle = any_oracle & ~full_oracle
                partial_actual = ordered_to_dense(mask.kv_num_blocks, mask.kv_indices)
                full_actual = ordered_to_dense(mask.full_kv_num_blocks, mask.full_kv_indices)
                torch.testing.assert_close(partial_actual, partial_oracle)
                torch.testing.assert_close(full_actual, full_oracle)


@requires_cuda
class FlexAttentionParityTests(unittest.TestCase):
    @staticmethod
    def dense_attention(q, k, v, document_ids, window, scale):
        length = q.shape[-2]
        positions = torch.arange(length, device=q.device)
        dense_mask = (
            (positions[:, None] >= positions)
            & (positions[:, None] - positions <= window)
            & (document_ids[:, None] == document_ids)
        )
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        scores.masked_fill_(~dense_mask, float("-inf"))
        return torch.matmul(torch.softmax(scores, dim=-1), v.float())

    def test_forward_and_gradients_match_dense_reference(self):
        from portable_attention import _build_document_block_mask
        from torch.nn.attention.flex_attention import flex_attention

        torch.manual_seed(7)
        length, heads, dim, window, scale = 256, 2, 16, 128, 0.12
        document_ids = torch.ones(length, device="cuda", dtype=torch.int64)
        document_ids[160:] = 2
        block_mask = _build_document_block_mask(document_ids, window)
        q = torch.randn(1, heads, length, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        q_ref = q.detach().clone().requires_grad_()
        k_ref = k.detach().clone().requires_grad_()
        v_ref = v.detach().clone().requires_grad_()

        actual = flex_attention(q, k, v, block_mask=block_mask, scale=scale).float()
        expected = self.dense_attention(q_ref, k_ref, v_ref, document_ids, window, scale)
        torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)
        actual.square().mean().backward()
        expected.square().mean().backward()
        for actual_grad, expected_grad in ((q.grad, q_ref.grad), (k.grad, k_ref.grad), (v.grad, v_ref.grad)):
            torch.testing.assert_close(actual_grad.float(), expected_grad.float(), atol=2e-3, rtol=8e-2)

    def test_paired_layout_restores_original_head_shape(self):
        from portable_attention import create_document_block_masks
        from torch.nn.attention.flex_attention import flex_attention

        torch.manual_seed(11)
        batch, length, heads, dim, window, scale = 1, 256, 6, 16, 128, 0.12
        seqlens = torch.tensor([0, 160, length, length], device="cuda", dtype=torch.int32)
        document_ids = torch.zeros(length, device="cuda", dtype=torch.int64)
        document_ids[160:] = 1
        document_ids = document_ids.repeat_interleave(2)
        (block_mask,) = create_document_block_masks(seqlens, length, (window,), paired=True)
        q = torch.randn(batch, length, heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        q_pair = q.view(batch, length, heads // 2, dim * 2).view(batch, 2 * length, heads // 2, dim)
        k_pair = k.view(batch, length, heads // 2, dim * 2).view(batch, 2 * length, heads // 2, dim)
        v_pair = v.reshape(batch, 2 * length, heads // 2, dim)

        actual = flex_attention(
            q_pair.transpose(1, 2),
            k_pair.transpose(1, 2),
            v_pair.transpose(1, 2),
            block_mask=block_mask,
            scale=scale,
        ).transpose(1, 2)[0].contiguous().view(batch, length, heads, dim)
        expected_pair = self.dense_attention(
            q_pair.transpose(1, 2),
            k_pair.transpose(1, 2),
            v_pair.transpose(1, 2),
            document_ids,
            window,
            scale,
        ).transpose(1, 2)[0].contiguous().view(batch, length, heads, dim)
        self.assertEqual(actual.shape, (batch, length, heads, dim))
        torch.testing.assert_close(actual.float(), expected_pair, atol=4e-2, rtol=4e-2)

    def test_mask_creation_and_flex_call_compile_fullgraph(self):
        require_triton()
        from portable_attention import create_document_block_masks
        from torch.nn.attention.flex_attention import flex_attention

        def attention(seqlens, q, k, v):
            (block_mask,) = create_document_block_masks(seqlens, q.shape[-2], (128,), paired=False)
            return flex_attention(q, k, v, block_mask=block_mask, scale=0.12)

        compiled = torch.compile(attention, dynamic=False, fullgraph=True)
        seqlens = torch.tensor([0, 160, 256, 256], device="cuda", dtype=torch.int32)
        q = torch.randn(1, 2, 256, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        output = compiled(seqlens, q, k, v)
        output.float().square().mean().backward()
        self.assertEqual(output.shape, q.shape)
        self.assertIsNotNone(q.grad)

    def test_mask_creation_has_no_dynamo_graph_break(self):
        from portable_attention import create_document_block_masks
        from torch.nn.attention.flex_attention import flex_attention

        def attention(seqlens, q, k, v):
            short_mask, long_mask = create_document_block_masks(
                seqlens,
                q.shape[-2],
                (128, 384),
                paired=False,
            )
            short = flex_attention(q, k, v, block_mask=short_mask, scale=0.12)
            long = flex_attention(q, k, v, block_mask=long_mask, scale=0.12)
            return short + long

        compiled = torch.compile(attention, backend="eager", dynamic=False, fullgraph=True)
        seqlens = torch.tensor([0, 289, 512, 512], device="cuda", dtype=torch.int32)
        q = torch.randn(1, 2, 512, 16, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        output = compiled(seqlens, q, k, v)
        self.assertEqual(output.shape, q.shape)

    def test_paired_reshape_and_backward_have_no_dynamo_graph_break(self):
        from portable_attention import create_document_block_masks
        from torch.nn.attention.flex_attention import flex_attention

        def paired_attention(seqlens, q, k, v):
            batch, length, heads, dim = q.shape
            (block_mask,) = create_document_block_masks(
                seqlens,
                length,
                (128,),
                paired=True,
            )
            q = q.view(batch, length, heads // 2, dim * 2).view(batch, 2 * length, heads // 2, dim)
            k = k.view(batch, length, heads // 2, dim * 2).view(batch, 2 * length, heads // 2, dim)
            v = v.reshape(batch, 2 * length, heads // 2, dim)
            return flex_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                block_mask=block_mask,
                scale=0.12,
            ).transpose(1, 2)[0].contiguous().view(batch, length, heads, dim)

        compiled = torch.compile(paired_attention, backend="eager", dynamic=False, fullgraph=True)
        seqlens = torch.tensor([0, 160, 256, 256], device="cuda", dtype=torch.int32)
        q = torch.randn(1, 256, 6, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        output = compiled(seqlens, q, k, v)
        output.float().square().mean().backward()
        self.assertEqual(output.shape, q.shape)
        self.assertIsNotNone(q.grad)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class KernelOptionTests(unittest.TestCase):
    def test_block_sizes_divide_the_mask_block_size(self):
        from portable_attention import BLOCK_SIZE, KERNEL_OPTIONS

        # The FlexAttention template asserts SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0 and
        # SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0; both sparse sizes are BLOCK_SIZE.
        for key in ("BLOCK_M", "BLOCK_N", "BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"):
            with self.subTest(key=key):
                value = KERNEL_OPTIONS[key]
                self.assertGreater(value, 0)
                self.assertLessEqual(value, BLOCK_SIZE)
                self.assertEqual(BLOCK_SIZE % value, 0)

    def test_tiles_fit_the_sm120_shared_memory_limit(self):
        from portable_attention import KERNEL_OPTIONS

        # Rough lower bound on the forward template's shared memory at head_dim 128:
        # a bf16 Q tile, pipelined bf16 K and V tiles, and an fp32 score tile. The
        # default 128/64/num_stages=2 config measured 180,248 B against SM120's
        # 101,376 B ceiling, so keep a wide margin here rather than a tight estimate.
        head_dim, sm120_limit = 128, 101_376
        block_m, block_n = KERNEL_OPTIONS["BLOCK_M"], KERNEL_OPTIONS["BLOCK_N"]
        stages = KERNEL_OPTIONS["num_stages"]
        estimate = (
            block_m * head_dim * 2
            + 2 * block_n * head_dim * 2 * stages
            + block_m * block_n * 4
        )
        self.assertLess(estimate, sm120_limit)


if __name__ == "__main__":
    unittest.main()

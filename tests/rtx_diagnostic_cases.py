"""GPU cases for diagnose_rtx5090.py. Imported only by isolated workers.

Controls and tracing patch this process only. Production modules are never edited.
Trace mode deliberately introduces graph boundaries, so compare it with the
unmodified fullgraph sampled/full cases before drawing conclusions.
"""
import hashlib
import math
import sys

import torch


def stats(label, tensor):
    """Scan even FP8 tensors with bounded FP32 scratch, and identify the first bad value."""
    if tensor is None:
        return
    if not isinstance(tensor, torch.Tensor):
        return
    values = tensor.detach().reshape(-1)
    low, high, bad = math.inf, -math.inf, 0
    for part in values.split(1 << 20):
        part = part.float()
        bad += int((~torch.isfinite(part)).sum().item())
        if part.numel():
            low = min(low, part.amin().item())
            high = max(high, part.amax().item())
    print(f"TENSOR {label} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
          f"min={low:.6g} max={high:.6g} nonfinite={bad}", flush=True)
    if bad:
        raise RuntimeError(f"First observed non-finite tensor: {label} ({bad} values)")


def compare(label, got, ref, tolerance):
    got, ref = got.detach().float(), ref.detach().float()
    if not torch.isfinite(ref).all().item() or not torch.isfinite(got).all().item():
        raise RuntimeError(f"{label}: non-finite comparison input")
    error = (got - ref).abs().amax().item()
    relative = error / max(ref.abs().amax().item(), 1e-12)
    print(f"PARITY {label} max_abs={error:.6g} max_relative={relative:.6g} limit={tolerance}", flush=True)
    if relative > tolerance:
        raise RuntimeError(f"{label}: relative error {relative:.6g} exceeds {tolerance}")


def sample_rows(rows):
    return sorted({0, max(0, rows // 2 - 64), max(0, rows - 128)})


def attempt(failures, label, operation):
    print(f"SUBCASE {label}", flush=True)
    try:
        operation()
        torch.cuda.synchronize()
        print(f"SUBCASE_RESULT {label} pass", flush=True)
    except Exception as exc:
        failures.append(f"{label}: {type(exc).__name__}: {exc}")
        print(f"SUBCASE_RESULT {failures[-1]}", flush=True)


def finish(failures):
    if failures:
        raise RuntimeError("; ".join(failures))


def gemm_case(tokens):
    """Separate accumulation precision, output dtype, layout and production shapes."""
    failures = []
    for label, k, n, x_scale, w_scale in (
        ("sampled_logits", 768, 10240, 100 / 448, 2 / 448),
        ("full_logits", 768, 50304, 100 / 448, 2 / 448),
        ("mlp_down", 2816, 768, 1 / 16, 1 / 128),
    ):
        x = torch.randn(tokens, k, device="cuda") * 0.3
        # Transpose (N,K), not (K,N): _scaled_mm requires a column-major right operand.
        w = torch.randn(n, k, device="cuda") * 0.03
        a = (x / x_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        b = (w / w_scale).clamp(-448, 448).to(torch.float8_e4m3fn).T
        a_s, b_s = (torch.tensor(s, device="cuda") for s in (x_scale, w_scale))
        b_ref = b.float() * b_s
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            for fast in (False, True):
                def check():
                    out = torch._scaled_mm(a, b, scale_a=a_s, scale_b=b_s,
                                           out_dtype=dtype, use_fast_accum=fast)
                    stats(f"{label} out={dtype} fast={fast}", out)
                    for start in sample_rows(tokens):
                        ref = (a[start:start + 128].float() * a_s) @ b_ref
                        compare(f"{label} rows={start}:{start + 128}", out[start:start + 128], ref,
                                0.12 if dtype == torch.float8_e4m3fn else 0.03)
                attempt(failures, f"{label}/{dtype}/fast={fast}", check)
        del x, w, a, b, b_ref
    finish(failures)


def mlp_case(tokens):
    """Fused FP8 up projection: row and transposed stores, eager versus compiled."""
    from track_1_short.perf.kernels.mlp import linear_relu_square, mlp_block_n, prime_stage_cache
    print(f"MLP tile width={mlp_block_n()}", flush=True)
    prime_stage_cache()
    x = torch.randn(tokens, 768, device="cuda", dtype=torch.bfloat16) * 0.3
    w = torch.randn(2816, 768, device="cuda", dtype=torch.bfloat16) * 0.03
    x_s = torch.tensor([1 / 16], device="cuda")
    w_s = (w.float().abs().amax() / 448).clamp_min(1e-12).reshape(1)
    a = (x.float() / x_s).clamp(-448, 448).to(torch.float8_e4m3fn)
    b = (w.float() / w_s).clamp(-448, 448).to(torch.float8_e4m3fn)
    dequant = x_s * w_s
    post_scale = torch.ones(1, device="cuda")
    amax = torch.zeros(1, device="cuda")

    def forward(x, w, a, b, dequant, post_scale, amax):
        _, _, row, transposed = linear_relu_square(
            x, w, a_f8=a, b_f8=b, dequant_scale_ptr=dequant, emit_f8=True, emit_t=True,
            post_scale=post_scale, post_amax=amax, store_pre=False, store_post_bf=False)
        return row, transposed

    failures = []
    for label, fn in (("eager", forward), ("compiled", torch.compile(forward, fullgraph=True))):
        def check():
            amax.zero_()
            row, transposed = fn(x, w, a, b, dequant, post_scale, amax)
            stats(f"{label} MLP post row", row)
            stats(f"{label} MLP post transposed", transposed)
            stats(f"{label} MLP post amax", amax)
            for start in sample_rows(tokens):
                pre = ((a[start:start + 128].float() @ b.float().T) * dequant).bfloat16()
                ref = pre.clamp_min(0).square().float().clamp(max=448)
                compare(f"{label} up projection rows={start}", row[start:start + 128], ref, 0.12)
                compare(f"{label} transpose rows={start}", transposed[:, start:start + 128].T,
                        row[start:start + 128], 0.0)
        attempt(failures, label, check)
    finish(failures)


def ce_case(tokens):
    """Known finite logits bypass GEMM: test CE itself at sampled and full widths."""
    from track_1_short.perf.kernels.cross_entropy import ce_fwd_bwd
    from track_1_short.perf.kernels.sampled_cross_entropy import sampled_ce_fwd_bwd
    failures = []
    rows = 128
    for width in (10240, 14336, 24576, 50304):
        for pattern in ("zeros", "random"):
            def check():
                logits = torch.zeros(rows, width, device="cuda") if pattern == "zeros" else torch.randn(rows, width, device="cuda")
                logits = logits.to(torch.float8_e4m3fn)
                targets = torch.arange(rows, device="cuda", dtype=torch.int64) % width
                losses = torch.empty(rows, device="cuda")
                grad = torch.empty(rows, width, device="cuda", dtype=torch.float8_e5m2)
                kernel = ce_fwd_bwd if width == 50304 else sampled_ce_fwd_bwd
                kernel(logits, targets, torch.ones(1, device="cuda"), torch.full_like(targets, -1),
                       torch.zeros(1, device="cuda"), losses, grad, rows, 1, 23., 5., 7.5, (0.75 / 8) / 448)
                stats(f"CE {width}/{pattern} losses", losses)
                stats(f"CE {width}/{pattern} logit gradients", grad)
                # The CUDA kernel caches sigmoid in FP16 before applying the softcap.
                scores = 23 * torch.sigmoid((logits.float() + 5) / 7.5).half().float()
                ref = torch.logsumexp(scores, dim=-1) - scores.gather(1, targets[:, None]).squeeze(1)
                compare(f"CE {width}/{pattern} loss", losses, ref, 0.002)
            attempt(failures, f"width={width}/{pattern}", check)
    finish(failures)


def capture_trainer():
    """Reuse the exact production initialization, stopping before its first warmup."""
    from pathlib import Path
    entry = sys.argv[0]
    try:
        sys.argv[0] = str(Path(__file__).resolve().parents[1] / "train_gpt_rtx.py")
        import train_gpt_rtx as module
    finally:
        sys.argv[0] = entry
    captured = []
    original = module.Trainer.__init__

    class Ready(Exception):
        pass

    def capture(self, *args, **kwargs):
        original(self, *args, **kwargs)
        captured.append(self)
        raise Ready()

    module.Trainer.__init__ = capture
    try:
        module.main()
    except Ready:
        pass
    finally:
        module.Trainer.__init__ = original
    if len(captured) != 1:
        raise RuntimeError("Failed to capture initialized production trainer")
    return module, captured[0]


def audit_logits(model, x, sampled):
    """Inspect the actual head inputs and the shared FP8 GEMM before either CE kernel."""
    from track_1_short.perf.kernels.cross_entropy import _ce_logit_gemm
    x = x.detach().reshape(-1, x.shape[-1])
    stats("loss input hidden", x)
    x_f8 = x.div(model.lm_head.x_s).to(torch.float8_e4m3fn)
    weights = sampled.rows.T if sampled is not None else model._lm_head_f8_col
    stats("loss input quantized hidden", x_f8)
    stats("loss input quantized weights", weights)
    logits = _ce_logit_gemm(x_f8, weights, model.lm_head.x_s, model.lm_head.w_s)
    stats("shared CE logit GEMM output", logits)
    ref = (x_f8[:128].float() * model.lm_head.x_s) @ (weights.float() * model.lm_head.w_s)
    compare("actual head logits, first 128 rows", logits[:128], ref, 0.12)


def install_trace(model):
    """Graph breaks allow synchronous tensor inspection; inner kernels remain compiled."""
    cls = type(model)
    compiled_mlp = torch.compile(cls._mlp, fullgraph=True, dynamic=False)

    @torch.compiler.disable
    def traced_mlp(self, x, w1, w2, fp8_args):
        slot = (w1.data_ptr() - self.mlp_bank.data_ptr()) // (2 * self.mlp_hdim * 768 * w1.element_size())
        stats(f"MLP slot {slot} input", x)
        if fp8_args:
            for index, tensor in enumerate(fp8_args):
                # Inspect primary caches and scales; transpose parity has its own kernel case.
                if index in (0, 1, 2, 3, 4, 5, 10, 12, 13, 14):
                    stats(f"MLP slot {slot} fp8 arg {index}", tensor)
        out = compiled_mlp(self, x, w1, w2, fp8_args)
        stats(f"MLP slot {slot} output", out)
        return out

    cls._mlp = traced_mlp
    # Bind each wrapper to its original method; no late-bound loop captures.
    for layer, attention in model.attn.items():
        compiled_attention = torch.compile(attention.forward, fullgraph=True, dynamic=False)

        def wrap(name, fn):
            @torch.compiler.disable
            def traced(x, args, qk, v, o, qkv_fp8=None):
                stats(f"attention L{name} input", x)
                if qkv_fp8 is not None:
                    for index in (0, 2, 3, 4, 5):
                        stats(f"attention L{name} fp8 arg {index}", qkv_fp8[index])
                out = fn(x, args, qk, v, o, qkv_fp8)
                stats(f"attention L{name} output", out)
                return out
            return traced
        attention.forward = wrap(layer, compiled_attention)
    compiled_loss = torch.compile(cls._loss, fullgraph=True, dynamic=False)

    @torch.compiler.disable
    def traced_loss(self, x, inputs, targets, mtp, prefix, sampled):
        audit_logits(self, x, sampled)
        out = compiled_loss(self, x, inputs, targets, mtp, prefix, sampled)
        stats("fused CE loss output", out)
        return out

    cls._loss = traced_loss


def forward_case(case, tokens):
    module, trainer = capture_trainer()
    model = trainer.uncompiled
    # Record small fingerprints to confirm comparable fresh inputs across processes.
    digest = hashlib.sha256(model.embed.weight[:16].detach().float().cpu().numpy().tobytes()).hexdigest()
    print(f"INITIAL_EMBED_SAMPLE_SHA256 {digest}", flush=True)
    manager = trainer.training_manager
    manager.advance_schedule(0)
    stage, _ = manager.schedule.lookup(0)
    staging = module.PinnedBatchStaging(tokens, module.microbatch_cu_seqlens_rows(tokens), trainer.device)
    batch = module.MicrobatchLoader(module.Hyperparameters().train_files, tokens, staging).next_step(1, stage.train_max_seq_len)[0]
    digest = hashlib.sha256(batch.inputs.cpu().numpy().tobytes()).hexdigest()
    print(f"INPUT_SHA256 {digest}", flush=True)
    sampled = None if case in ("eval", "full") else trainer.sampled.prepare(0, batch.targets_cpu, model.lm_head_f8_col)
    if sampled is not None:
        width = sampled.rows.shape[0]
        positions = sampled.target_pos[:tokens]
        if not ((positions >= 0) & (positions < width)).all().item():
            raise RuntimeError("Sampled target positions are outside the candidate vocabulary")
        candidates = trainer.sampled.candidates[:width]
        if not torch.equal(candidates[positions], batch.targets):
            raise RuntimeError("Sampled target positions do not map back to the target tokens")
        expected = model.lm_head_f8_col.T.view(torch.uint8).index_select(0, candidates)
        if not torch.equal(sampled.rows.view(torch.uint8), expected):
            raise RuntimeError("Sampled head rows differ from the source FP8 cache")
        if not torch.equal(sampled.rows_t.view(torch.uint8), sampled.rows.view(torch.uint8).T):
            raise RuntimeError("Sampled head transposed cache differs from its row cache")
        del expected
        print(f"CANDIDATES width={width} target_mapping=pass cached_rows=pass transpose=pass", flush=True)
    cfg = manager.get_forward_args(sampled)
    masks = module.build_window_masks(batch.cum_seqlens, tokens, cfg.ws_short, cfg.ws_long)
    print(f"FORWARD mode={case} mtp={cfg.mtp_weights.tolist()} prefix={cfg.prefix_weight.item()}", flush=True)

    if case == "bf16_mlp_control":
        def plain_mlp(self, x, w1, w2, fp8_args):
            out = torch.relu(x @ w1.T).square() @ w2
            # Preserve the post-lambda that the FP8 down projection normally folds in.
            return out * fp8_args[16] if fp8_args is not None and len(fp8_args) == 17 else out
        type(model)._mlp = plain_mlp
        print("CONTROL: replace only MLP arithmetic with differentiable BF16 matmuls.", flush=True)
    elif case == "bf16_logits_control":
        import track_1_short.perf.kernels.cross_entropy as ce
        import track_1_short.perf.kernels.sampled_cross_entropy as sampled_ce

        def bf16_logits(x_f8, weights, x_s, w_s):
            out = torch._scaled_mm(x_f8, weights, out_dtype=torch.bfloat16,
                                   scale_a=x_f8.new_tensor(x_s, dtype=torch.float32),
                                   scale_b=x_f8.new_tensor(w_s, dtype=torch.float32), use_fast_accum=True)
            return out.clamp(-448, 448).to(torch.float8_e4m3fn)
        ce._ce_logit_gemm = sampled_ce._ce_logit_gemm = bf16_logits
        print("CONTROL: BF16 GEMM output, then clamp and convert to FP8; CE stays unchanged.", flush=True)
    elif case == "trace":
        install_trace(model)
        print("TRACE: segmented compilation, not the identical fullgraph execution.", flush=True)

    compiled = torch.compile(model, fullgraph=case != "trace", dynamic=False)
    model.train(case != "eval")
    kwargs = {"window_masks": masks}
    if case == "eval":
        model.complete_yarn_tables()
    else:
        kwargs.update(ngram_sink=trainer.ngram.grad_sink(tokens), value_embed_grad=trainer.value_embeds.grad_accum)
    # Training controls remain grad-enabled so AOT uses the training variants.
    with torch.set_grad_enabled(case != "eval"):
        losses = compiled(batch.inputs, batch.targets, batch.cum_seqlens, batch.ngram_ids, cfg, **kwargs)
        stats(f"{case} per-token loss", losses)
        print(f"LOSS {case} mean={losses.mean().item():.8f}", flush=True)
        if case in ("sampled", "full"):
            losses.sum().backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    stats(f"gradient {name}", parameter.grad)
            stats("gradient ngram sink", kwargs["ngram_sink"].grad)
            stats("gradient value embeds", trainer.value_embeds.grad_accum)
            print("BACKWARD finite=yes (no optimizer update)", flush=True)


def run_case(case, tokens):
    if case == "gemm":
        gemm_case(tokens)
    elif case == "mlp":
        mlp_case(tokens)
    elif case == "ce":
        ce_case(tokens)
    else:
        forward_case(case, tokens)

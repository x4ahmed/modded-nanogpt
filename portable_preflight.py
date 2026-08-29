"""RTX-only checks used by the executable two-GPU preflight."""

from __future__ import annotations

import math


GIB = 1024**3


def memory_headroom_bytes(total_memory: int, peak_reserved: int) -> int:
    return total_memory - peak_reserved


def memory_gate_passes(total_memory: int, peak_reserved: int, min_headroom_gib: float) -> bool:
    if not math.isfinite(min_headroom_gib) or min_headroom_gib < 0:
        raise ValueError("minimum headroom must be finite and nonnegative")
    return memory_headroom_bytes(total_memory, peak_reserved) >= min_headroom_gib * GIB


def _ce_reference(torch, logits, targets, prefix_targets, prefix_weight, *, A, B, C):
    # Match the CUDA kernel's tanh sigmoid and BF16 shared-memory staging.
    sigmoid_u = (0.5 + 0.5 * torch.tanh(((logits.float() + B) / C) * 0.5))
    sigmoid_u = sigmoid_u.to(torch.bfloat16).float()
    z = A * sigmoid_u
    logsumexp = torch.logsumexp(z, dim=-1)

    row = torch.arange(logits.shape[0], device=logits.device)
    losses = logsumexp - z[row, targets]
    prefix_valid = prefix_targets >= 0
    safe_prefix = prefix_targets.clamp_min(0)
    losses = losses + prefix_valid * prefix_weight * (logsumexp - z[row, safe_prefix])

    target_weight = torch.zeros_like(z)
    target_weight.scatter_add_(1, targets[:, None], torch.ones_like(targets[:, None], dtype=z.dtype))
    target_weight.scatter_add_(
        1,
        safe_prefix[:, None],
        (prefix_valid * prefix_weight).to(z.dtype)[:, None],
    )
    total_weight = 1.0 + prefix_valid.to(z.dtype) * prefix_weight
    grad_z = total_weight[:, None] * torch.softmax(z, dim=-1) - target_weight
    grad_logits = (A / C) * grad_z * sigmoid_u * (1.0 - sigmoid_u)
    return losses, grad_logits.to(torch.float8_e5m2)


def run_ce_preflight(report) -> None:
    """Compile, configure, launch, synchronize, and compare CE as distinct stages."""
    import torch
    from triton_kernels import (
        CE_KERNEL_COMPUTE_CAPABILITY,
        CE_KERNEL_DYNAMIC_SHARED_BYTES,
        CE_KERNEL_STATIC_SHARED_BYTES_MIN,
        CE_KERNEL_TOTAL_SHARED_BYTES_MIN,
        CE_KERNEL_VOCAB_SIZE,
        ce_fwd_bwd,
        compile_ce_kernel,
        configure_ce_kernel,
    )

    def stage(name, operation):
        try:
            return operation()
        except Exception as exc:
            report(f"{name} fail error={type(exc).__name__}:{exc}")
            raise

    report("CE_MODULE_IMPORT pass")
    stage("CE_COMPILE", compile_ce_kernel)
    report(f"CE_COMPILE pass target={CE_KERNEL_COMPUTE_CAPABILITY}")

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    optin_limit = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
    stage("CE_SHARED_CONFIG", configure_ce_kernel)
    report(
        "CE_SHARED_CONFIG pass "
        f"dynamic_bytes={CE_KERNEL_DYNAMIC_SHARED_BYTES} "
        f"static_bytes_min={CE_KERNEL_STATIC_SHARED_BYTES_MIN} "
        f"total_bytes_min={CE_KERNEL_TOTAL_SHARED_BYTES_MIN} "
        f"device_optin_limit_bytes={optin_limit}"
    )
    if optin_limit < CE_KERNEL_TOTAL_SHARED_BYTES_MIN:
        raise RuntimeError("CE kernel exceeds the device per-block shared-memory opt-in limit")

    n_rows = 2
    logits = torch.linspace(
        -2.0,
        2.0,
        CE_KERNEL_VOCAB_SIZE,
        device="cuda",
        dtype=torch.float32,
    ).to(torch.bfloat16).repeat(n_rows, 1).contiguous()
    targets = torch.tensor([17, CE_KERNEL_VOCAB_SIZE - 19], device="cuda", dtype=torch.int64)
    mtp_weights = torch.ones(1, device="cuda", dtype=torch.float32)
    # Row 0 deliberately duplicates its next-token target; row 1 disables prefix loss.
    prefix_targets = torch.tensor([17, -1], device="cuda", dtype=torch.int64)
    prefix_weight = torch.tensor([0.25], device="cuda", dtype=torch.float32)
    losses = torch.empty(n_rows, device="cuda", dtype=torch.float32)
    grad_input = torch.empty_like(logits, dtype=torch.float8_e5m2)
    A, B, C = 23.0, 5.0, 7.5

    stage(
        "CE_LAUNCH",
        lambda: ce_fwd_bwd(
            logits,
            targets,
            mtp_weights,
            prefix_targets,
            prefix_weight,
            losses,
            grad_input,
            n_rows,
            1,
            A,
            B,
            C,
            1.0,
            1.0,
        ),
    )
    report("CE_LAUNCH pass enqueue_only=true")
    stage("CE_SYNC", torch.cuda.synchronize)
    report("CE_SYNC pass")

    def parity_metrics():
        ref_losses, ref_grad = _ce_reference(
            torch,
            logits,
            targets,
            prefix_targets,
            prefix_weight,
            A=A,
            B=B,
            C=C,
        )
        loss_abs = (losses - ref_losses).abs().max().item()
        actual_grad = grad_input.float().flatten()
        expected_grad = ref_grad.float().flatten()
        grad_rel_l2 = ((actual_grad - expected_grad).norm() / expected_grad.norm()).item()
        grad_cosine = (
            torch.dot(actual_grad, expected_grad) / (actual_grad.norm() * expected_grad.norm())
        ).item()
        grad_exact = (actual_grad == expected_grad).float().mean().item()
        metrics = (loss_abs, grad_rel_l2, grad_cosine, grad_exact)
        if not all(math.isfinite(metric) for metric in metrics) or (
            loss_abs > 1e-2
            or grad_rel_l2 > 0.10
            or grad_cosine < 0.999
            or grad_exact < 0.98
        ):
            raise RuntimeError(
                "CE parity failed: "
                f"loss_abs={loss_abs:.6g} grad_rel_l2={grad_rel_l2:.6g} "
                f"grad_cosine={grad_cosine:.6g} grad_exact={grad_exact:.6g}"
            )
        return loss_abs, grad_rel_l2, grad_cosine, grad_exact

    loss_abs, grad_rel_l2, grad_cosine, grad_exact = stage("CE_PARITY", parity_metrics)
    report(
        "CE_PARITY pass "
        f"loss_max_abs={loss_abs:.6g} grad_rel_l2={grad_rel_l2:.6g} "
        f"grad_cosine={grad_cosine:.6g} grad_exact_fraction={grad_exact:.6g}"
    )

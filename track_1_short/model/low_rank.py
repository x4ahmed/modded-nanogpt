"""Rank-384 down-projection for the second MLP at position 8 only."""
import math

import torch
import torch.nn.functional as F
from torch import nn


PARALLEL_DOWN_RANK = 384


class LowRankParallelDown(nn.Module):
    """relu(x W_up.T)^2 A B.T, with A[H,r] and B[D,r].

    A starts at zero, preserving the baseline's zero-output MLP initialization.
    B is nonzero so A can learn immediately. A private RNG scope preserves all
    baseline initialization draws and the RNG state used by subsequent code.
    Each factor is a single replicated ANVIL matrix, including on eight GPUs.
    """
    def __init__(self, hidden_dim, model_dim, rank=PARALLEL_DOWN_RANK, *, device=None):
        super().__init__()
        if not 0 < rank < model_dim:
            raise ValueError("rank must be positive and smaller than model_dim")
        self.rank = rank
        self.a = nn.Parameter(torch.zeros(1, hidden_dim, rank, device=device))
        self.b = nn.Parameter(torch.empty(1, model_dim, rank, device=device))
        self.a.reshape = tuple(self.a.shape)
        self.b.reshape = tuple(self.b.shape)
        devices = [self.b.device.index] if self.b.is_cuda else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            self.b.uniform_(-math.sqrt(3 / rank), math.sqrt(3 / rank))
        for name, weight in (("a", self.a), ("b", self.b)):
            _, rows, cols = weight.shape
            # Same 64x64 tile count as quantize_mlp_weights_dual, without
            # importing Triton during CPU initialization/reference tests.
            tiles = math.ceil(rows / 64) * math.ceil(cols / 64)
            self.register_buffer(f"{name}_row", torch.empty_like(weight, dtype=torch.float8_e4m3fn), persistent=False)
            self.register_buffer(f"{name}_col", torch.empty(1, cols, rows, device=device,
                                 dtype=torch.float8_e4m3fn).transpose(1, 2), persistent=False)
            self.register_buffer(f"{name}_scale", torch.ones(1, device=device), persistent=False)
            self.register_buffer(f"{name}_amax", torch.zeros(1, tiles, device=device), persistent=False)
        self.register_buffer("a_scale_folded", torch.ones(1, device=device), persistent=False)
        self.register_buffer("a_dq_bwd", torch.ones(1, device=device), persistent=False)
        self.register_buffer("a_dq_bwd_folded", torch.ones(1, device=device), persistent=False)

    def reference(self, x, up):
        """Differentiable reference for CPU tests, not a production fallback."""
        post = F.relu(F.linear(x, up)).square()
        return F.linear(post @ self.a[0].to(x.dtype), self.b[0].to(x.dtype))

    def forward(self, middle, grad_scale=None):
        if grad_scale is None:  # BF16 validation
            return F.linear(middle, self.b[0].to(middle.dtype))
        from track_1_short.perf.kernels.mlp import FP8LowRankReadout
        # Current (not delayed) scale handles the exactly-zero first forward
        # and the first nonzero A update without saturation of the bottleneck.
        scale = (middle.detach().float().abs().amax() / 448).clamp_min(1e-12)
        return FP8LowRankReadout.apply(middle, self.b[0], self.b_row[0],
                                      self.b_col[0], self.b_scale[0], scale, grad_scale)

    def mlp(self, x, up, fp8_args=None):
        """Production path; only the down projection differs from the dense MLP."""
        from track_1_short.perf.kernels.mlp import FusedLinearReLUSquareFunction
        if fp8_args is None:
            middle = FusedLinearReLUSquareFunction.apply(x, up, self.a[0])
            return self(middle)
        folded = len(fp8_args) == 17
        args = (fp8_args[:3] +
                (self.a_col[0], self.a_scale_folded[0] if folded else self.a_scale[0]) +
                fp8_args[5:7] +
                (self.a_row[0], self.a_dq_bwd_folded[0] if folded else self.a_dq_bwd[0]) +
                fp8_args[9:])
        middle = FusedLinearReLUSquareFunction.apply(x, up, self.a[0], *args)
        return self(middle, fp8_args[13])

    @torch.no_grad()
    def refresh(self, *, exact, post_lambda, grad_scale):
        from track_1_short.perf.kernels.mlp import quantize_mlp_weights_dual
        for name in ("a", "b"):
            weight = getattr(self, name)
            scales, amax = getattr(self, f"{name}_scale"), getattr(self, f"{name}_amax")
            if exact:
                scales.copy_((weight.flatten(1).abs().amax(1).clamp_min(1e-12) / 448).float())
            quantize_mlp_weights_dual(weight, scales, amax, row=getattr(self, f"{name}_row"),
                                     col_t=getattr(self, f"{name}_col").transpose(1, 2), update_scales=not exact)
        torch.mul(self.a_scale, post_lambda, out=self.a_scale_folded)
        torch.mul(self.a_scale, grad_scale, out=self.a_dq_bwd)
        torch.mul(self.a_dq_bwd, post_lambda, out=self.a_dq_bwd_folded)

    @torch.no_grad()
    def rearm(self):
        self.a_amax.zero_()
        self.b_amax.zero_()

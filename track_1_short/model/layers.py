"""Small building blocks shared across the model."""
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


@torch.library.custom_op("nanogpt::token_shift", mutates_args=())
def token_shift(x: Tensor, w: Tensor) -> Tensor:
    """x + w * (the previous position's x) along the sequence (dim 1); the first position sees zeros.

    An opaque op with a hand-written backward. Written as plain ops inside the compiled model, the
    generated backward gave w a non-finite gradient on the RTX 5090 while every input was finite and
    small; as this op it trains (tests/debug_layer_smear.py). The price: the compiler cannot fuse it with
    its neighbours. Like the input smear it does not stop at document boundaries.
    """
    return torch.addcmul(x, F.pad(x[:, :-1], (0, 0, 1, 0)), w)


@token_shift.register_fake
def _(x, w):
    return torch.empty_like(x)


@torch.library.custom_op("nanogpt::token_shift_backward", mutates_args=())
def _token_shift_backward(grad: Tensor, x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    grad_x = grad.clone()
    grad_x[:, :-1].addcmul_(grad[:, 1:], w)
    acc = torch.promote_types(grad.dtype, torch.float32)  # bf16 products, summed in fp32
    grad_w = (grad[:, 1:] * x[:, :-1]).sum(dim=(0, 1), dtype=acc)
    return grad_x, grad_w.to(w.dtype)


@_token_shift_backward.register_fake
def _(grad, x, w):
    return torch.empty_like(grad), torch.empty_like(w)


def _token_shift_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _token_shift_grad(ctx, grad):
    return _token_shift_backward(grad, *ctx.saved_tensors)


token_shift.register_autograd(_token_shift_grad, setup_context=_token_shift_setup)


class CastedLinearT(nn.Module):
    """
    Linear layer with transposed weight storage (in_features, out_features) which
    addresses the slow kernel that was used for gradient accumulation. @chrisjmccormick

    Its forward is the bf16 path (validation). Training reads the weight through the fp8 loss in
    perf/kernels/cross_entropy.py, which uses the fp8 scales x_s, w_s and grad_s stored here.
    """
    def __init__(self, in_features: int, out_features: int, x_s: float, w_s: float, grad_s: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

        self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.bfloat16))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.weight) # @Grad62304977 and others

    def forward(self, x: Tensor):
        return x @ self.weight.type_as(x)


def next_multiple_of_n(v: float | int, *, n: int):
    return math.ceil(v / n) * n

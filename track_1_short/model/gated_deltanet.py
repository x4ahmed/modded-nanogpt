"""Gated DeltaNet token mixer (Yang, Kautz, Hatamizadeh, arXiv:2412.06464) for one attention layer.

What: the layer's softmax attention is replaced by the gated delta rule, a linear-attention recurrence
with a per-head memory S [K, V] that is reset at every document (segment) boundary:

    S_t = a_t * S_{t-1} + beta_t * k_t (v_t - a_t * S_{t-1}^T k_t)^T,     o_t = S_t^T q_t / sqrt(K)

a_t = exp(g_t) in (0, 1] is a learned per-head decay, beta_t in (0, 1) a learned per-head write
strength, q and k are L2-normalized. Unlike the sliding window it replaces, the memory spans the whole
document. As in the paper's layer: a short causal depthwise conv + SiLU on q, k, v, a per-head
RMS-normed output times a SiLU output gate. Kimi Delta Attention differs mainly by a per-channel decay.

The q/k/v/o projections are the replaced layer's own bank slices (qk_bank / vo_bank rows, so ANVIL keeps
training them), applied in bf16 (no fp8 for this layer). The extra parameters (conv, decay/beta
projection, output gate, output norm) are small and trained with Adam.

Kernel: flash-linear-attention's chunked Triton kernel (`fla.ops.gated_delta_rule`), wrapped as a
torch custom op so the compiled model stays one graph. The backward recomputes the forward through
fla's own autograd (one extra layer forward). `gated_delta_reference` is the same recurrence in plain
torch, for CPU tests and a GPU parity check.
"""
import math

import torch
import torch.nn.functional as F
from torch import Tensor

CONV_SIZE = 4


def _fla_gated_delta(q, k, v, g, beta, cu_seqlens):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    # The packed table pads with repeated end offsets (zero-length segments): drop them for fla.
    cu = torch.unique_consecutive(cu_seqlens.long())
    o, _ = chunk_gated_delta_rule(q[None], k[None], v[None], g[None], beta[None],
                                  cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
    return o[0]


@torch.library.custom_op("nanogpt::gated_delta", mutates_args=())
def gated_delta(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, cu_seqlens: Tensor) -> Tensor:
    """q, k [T, H, K]; v [T, H, V]; g [T, H] fp32 log-decay; beta [T, H]; cu_seqlens padded int32 offsets."""
    return _fla_gated_delta(q, k, v, g, beta, cu_seqlens)


@gated_delta.register_fake
def _(q, k, v, g, beta, cu_seqlens):
    return torch.empty_like(v)


@torch.library.custom_op("nanogpt::gated_delta_bwd", mutates_args=())
def gated_delta_bwd(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, cu_seqlens: Tensor,
                    grad_o: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    # A custom op runs below the autograd dispatch keys; re-enable them to differentiate through fla.
    keys = torch._C.DispatchKey
    exclude = (torch._C._dispatch_tls_local_exclude_set().remove(keys.AutogradFunctionality)
               .remove(keys.AutogradOther).remove(keys.AutogradNestedTensor))
    with torch._C._ForceDispatchKeyGuard(torch._C._dispatch_tls_local_include_set(), exclude), torch.enable_grad():
        inputs = [t.detach().requires_grad_() for t in (q, k, v, g, beta)]
        o = _fla_gated_delta(*inputs, cu_seqlens)
        grads = torch.autograd.grad(o, inputs, grad_o)
    return tuple(gr.contiguous() for gr in grads)


@gated_delta_bwd.register_fake
def _(q, k, v, g, beta, cu_seqlens, grad_o):
    return tuple(torch.empty_like(t) for t in (q, k, v, g, beta))


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad_o):
    q, k, v, g, beta, cu_seqlens = ctx.saved_tensors
    dq, dk, dv, dg, dbeta = gated_delta_bwd(q, k, v, g, beta, cu_seqlens, grad_o.contiguous())
    return dq, dk, dv, dg, dbeta, None


gated_delta.register_autograd(_backward, setup_context=_setup_context)


def gated_delta_reference(q, k, v, g, beta, cu_seqlens):
    """The recurrence above, token by token in fp32 (slow; tests only). Same arguments as gated_delta."""
    T, H, K = q.shape
    q = F.normalize(q.float(), dim=-1) / math.sqrt(K)
    k = F.normalize(k.float(), dim=-1)
    v, g, beta = v.float(), g.float(), beta.float()
    out = torch.zeros(T, H, v.shape[-1])
    bounds = torch.unique_consecutive(cu_seqlens.long()).tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        state = torch.zeros(H, K, v.shape[-1])
        for t in range(start, end):
            state = state * g[t].exp()[:, None, None]
            err = v[t] - torch.einsum("hkv,hk->hv", state, k[t])
            state = state + beta[t][:, None, None] * k[t][:, :, None] * err[:, None, :]
            out[t] = torch.einsum("hkv,hk->hv", state, q[t])
    return out.to(v.dtype)


def segment_offsets(cu_seqlens: Tensor, num_tokens: int) -> Tensor:
    """Each token's position within its segment, from the padded cu_seqlens table."""
    pos = torch.arange(num_tokens, device=cu_seqlens.device)
    cu = cu_seqlens.long()
    start = cu[torch.searchsorted(cu, pos, right=True) - 1]
    return pos - start


def causal_conv(x: Tensor, weight: Tensor, offsets: Tensor) -> Tensor:
    """Depthwise causal conv over tokens, x [T, C], weight [C, CONV_SIZE] (last tap = current token);
    taps that would reach into the previous segment are dropped."""
    T = x.shape[0]
    w = weight.type_as(x)
    y = x * w[:, -1]
    for lag in range(1, weight.shape[1]):
        shifted = F.pad(x, (0, 0, lag, 0))[:T]
        y = y + shifted * w[:, -1 - lag] * (offsets >= lag).to(x.dtype)[:, None]
    return y


def init_decay(num_heads: int) -> tuple[Tensor, Tensor]:
    """(A_log, dt_bias) as in the paper's layer, but deterministic (spread over heads, no RNG draws):
    A in [1, 16], dt in [1e-3, 0.1]; decay g = -exp(A_log) * softplus(a + dt_bias)."""
    a = torch.linspace(1.0, 16.0, num_heads)
    dt = torch.exp(torch.linspace(math.log(1e-3), math.log(0.1), num_heads))
    return a.log(), dt + torch.log(-torch.expm1(-dt))  # inverse softplus

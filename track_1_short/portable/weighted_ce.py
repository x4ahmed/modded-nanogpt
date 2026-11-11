"""The stratified-pruning training loss (portable/vocab_prune.py): the record's sampled-softmax loss with
one log weight per candidate column.

The record's CE kernel computes lse = log(sum_c exp(z_c)) over its candidate columns. Here column c also
carries col_logw[c]: lse = log(sum_c w_c exp(z_c)) and the gradient of column c is w_c exp(z_c - lse)
(times the record's softcap chain factor), with w_c = 1 for exact columns and w_c = rest / tail for the
uniformly drawn tail. The kernel is the record's source with four text substitutions
(vocab_prune.weighted_kernel_source); GEMMs, scales and the gradient densify are the record's.
"""
import torch

from track_1_short.perf.kernels.cross_entropy import (
    CE_KERNEL_BLOCK_SIZE,
    CE_KERNEL_SOURCE,
    CUDA_INCLUDE_DIRS,
    SOFTCAP_A,
    SOFTCAP_B,
    SOFTCAP_C,
    _ce_backward_gemms,
    _ce_logit_gemm,
)
from track_1_short.perf.kernels.sampled_cross_entropy import sampled_densify
from track_1_short.portable import config as rtx
from track_1_short.portable.vocab_prune import ranked_widths_in_use, weighted_kernel_source
from track_1_short.sampled_softmax import ALL_CANDIDATE_COUNTS

WEIGHTED_CE_SOURCE = weighted_kernel_source(CE_KERNEL_SOURCE)


def compile_weighted_ce_kernel(width: int):
    """As cross_entropy.compile_ce_kernel, from the weighted source."""
    decls = f"\nconstexpr int VOCAB_SIZE = {width};\nconstexpr int BLOCK_SIZE = {CE_KERNEL_BLOCK_SIZE};\n"
    kernel = torch.cuda._compile_kernel(
        decls + WEIGHTED_CE_SOURCE,
        "ce_fwd_bwd_kernel",
        compute_capability="90",
        cuda_include_dirs=CUDA_INCLUDE_DIRS,
        nvcc_options=["-lineinfo", "--use_fast_math"],
    )
    kernel.set_shared_memory_config(width * 2)
    return kernel


# Compiled at import (off the clock), only when a stratified run asks for them.
WEIGHTED_CE_KERNELS = ({w: compile_weighted_ce_kernel(w) for w in ranked_widths_in_use(
    ALL_CANDIDATE_COUNTS, rtx.VOCAB_PRUNE_MASS, rtx.VOCAB_PRUNE_TAIL, 50304)}
    if rtx.PORTABLE and rtx.VOCAB_PRUNE_MASS and rtx.VOCAB_PRUNE_TAIL else {})


@torch.library.custom_op("nanogpt::weighted_sampled_ce_fwd_bwd", mutates_args={"losses", "grad_input"})
def weighted_sampled_ce_fwd_bwd(
    logits: torch.Tensor,
    target_pos: torch.Tensor,
    mtp_weights: torch.Tensor,
    prefix_pos: torch.Tensor,
    prefix_weight: torch.Tensor,
    col_logw: torch.Tensor,
    losses: torch.Tensor,
    grad_input: torch.Tensor,
    n_rows: int,
    n_predict: int,
    A: float,
    B: float,
    C: float,
    grad_s: float,
) -> None:
    P = logits.shape[1]
    WEIGHTED_CE_KERNELS[P](
        (n_rows, 1, 1),
        (CE_KERNEL_BLOCK_SIZE, 1, 1),
        (logits, target_pos, mtp_weights, prefix_pos, prefix_weight, col_logw, losses, grad_input,
         n_rows, n_predict, A, B, C, grad_s),
        shared_mem=P * 2,
    )


class WeightedSampledSoftcappedCrossEntropy(torch.autograd.Function):
    """SampledSoftcappedCrossEntropy (perf/kernels/sampled_cross_entropy.py) plus col_logw."""
    @staticmethod
    def forward(ctx, x, mtp_weights, target_pos, prefix_pos, prefix_weight, lm_head_weight,
                rows, rows_t, vocab_pos, col_logw, x_s, w_s, grad_s):
        n_rows = x.shape[0]
        P = rows.shape[0]
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        logits = _ce_logit_gemm(x_f8, rows.T, x_s, w_s)
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        grad_input = torch.empty((n_rows, P), dtype=torch.float8_e5m2, device=logits.device)
        weighted_sampled_ce_fwd_bwd(logits.contiguous(), target_pos, mtp_weights.contiguous(), prefix_pos,
                                    prefix_weight.reshape(1).to(torch.float32).contiguous(), col_logw, losses,
                                    grad_input, n_rows, mtp_weights.shape[0], SOFTCAP_A, SOFTCAP_B, SOFTCAP_C, grad_s)
        ctx.save_for_backward(x_f8, rows_t, grad_input, vocab_pos)
        ctx.scales = (x_s, w_s, grad_s)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        x_f8, rows_t, grad_input, vocab_pos = ctx.saved_tensors
        grad_x, grad_w_c = _ce_backward_gemms(grad_input, rows_t, x_f8, *ctx.scales)
        grad_w = torch.empty((x_f8.shape[1], vocab_pos.numel()), dtype=grad_w_c.dtype, device=grad_w_c.device)
        sampled_densify(grad_w_c.contiguous(), vocab_pos, grad_w)
        return grad_x, None, None, None, None, grad_w, None, None, None, None, None, None, None

"""Symmetric polynomial products for paired Gram Newton--Schulz.

Only for matrices that commute in exact arithmetic. Store one lower triangle
and mirror it, including diagonal tiles: no duplicate stores or diagonal races.
FP32 accumulation/epilogue, one BF16 store. No TMA or Hopper-only instructions.
Inputs and output must not alias. Coefficients are compile-time constants.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gram_product(A, B, C, Out, N: tl.constexpr,
                  asb: tl.constexpr, asr: tl.constexpr, asc: tl.constexpr,
                  bsb: tl.constexpr, bsr: tl.constexpr, bsc: tl.constexpr,
                  csb: tl.constexpr, csr: tl.constexpr, csc: tl.constexpr,
                  osb: tl.constexpr, osr: tl.constexpr, osc: tl.constexpr,
                  ALPHA: tl.constexpr, BETA: tl.constexpr, DIAGONAL: tl.constexpr,
                  BLOCK: tl.constexpr, BK: tl.constexpr):
    batch = tl.program_id(1)
    tiles = tl.cdiv(N, BLOCK)
    row_tile, col_tile = tl.program_id(0) // tiles, tl.program_id(0) % tiles
    if col_tile > row_tile:
        return
    rows = row_tile * BLOCK + tl.arange(0, BLOCK)
    cols = col_tile * BLOCK + tl.arange(0, BLOCK)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    for start in range(tl.cdiv(N, BK)):
        k = start * BK + ks
        left = tl.load(A + batch * asb + rows[:, None] * asr + k[None, :] * asc,
                       (rows[:, None] < N) & (k[None, :] < N), 0)
        right = tl.load(B + batch * bsb + k[:, None] * bsr + cols[None, :] * bsc,
                        (k[:, None] < N) & (cols[None, :] < N), 0)
        acc = tl.dot(left, right, acc)
    acc = acc * ALPHA
    valid = (rows[:, None] < N) & (cols[None, :] < N)
    if BETA != 0.:
        addend = tl.load(C + batch * csb + rows[:, None] * csr + cols[None, :] * csc,
                         valid, 0).to(tl.float32)
        acc = acc + BETA * addend
    if DIAGONAL != 0.:
        acc = acc + tl.where(rows[:, None] == cols[None, :], DIAGONAL, 0.)
    value = acc.to(Out.dtype.element_ty)
    lower = rows[:, None] >= cols[None, :]
    tl.store(Out + batch * osb + rows[:, None] * osr + cols[None, :] * osc,
             value, valid & lower)
    # Transposed addresses, same values. The strict mask excludes the diagonal.
    tl.store(Out + batch * osb + cols[None, :] * osr + rows[:, None] * osc,
             value, valid & (rows[:, None] > cols[None, :]))


def sym_product(A, B, C, *, alpha, beta, diagonal, out):
    """Mirror lower(alpha*A@B + beta*C + diagonal*I), with a fused epilogue."""
    assert A.ndim in (2, 3) and A.size(-2) == A.size(-1)
    assert A.shape == B.shape == C.shape == out.shape
    assert A.dtype == B.dtype == C.dtype == out.dtype == torch.bfloat16
    assert A.is_cuda and A.device == B.device == C.device == out.device
    strides = lambda t: (t.stride(0) if t.ndim == 3 else 0, t.stride(-2), t.stride(-1))
    # Conservative pointer-kernel configuration. Offline Triton 3.4 compilation
    # used 16,640 bytes of shared memory on SM120 and 66,560 on SM90.
    # Actual launch resources and timing must be checked on the target GPU.
    n = A.size(-1)
    _gram_product[(triton.cdiv(n, 128) ** 2, A.size(0) if A.ndim == 3 else 1)](
        A, B, C, out, n, *strides(A), *strides(B), *strides(C), *strides(out),
        alpha, beta, diagonal, 128, 32, num_warps=8, num_stages=3,
    )
    return out

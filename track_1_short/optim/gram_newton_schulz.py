"""Paired Gram Newton--Schulz for ANVIL's tall MLP matrices.

Keep ANVIL's input normalization and six maps. For a pair of maps, write
Q1 = a1 I + b1 R + c1 R^2, R1 = Q1 R Q1, Q2 = a2 I + b2 R1 + c2 R1^2,
then X <- X (Q1 Q2). Rebuild R = X.T X between pairs to limit rounding drift.
This is algebraically equivalent to two ordinary quintics, not bitwise equivalent.

Algorithm reference: https://github.com/Dao-AILab/gram-newton-schulz
This implementation retains BF16 and ANVIL's coefficients, rather than that
package's FP16 normalization/default coefficients. GPU validation is required.
Kernel callbacks keep this algebra independently testable without CUDA/Triton.
"""
import torch


def gram_pair_cascade(X, gram, coefficients, *, gram_out, sym_product_out):
    """Consume normalized tall X and its already-computed Gram; return BF16 update.

    X and gram are scratch buffers, not optimizer state. sym_product_out computes
    the lower triangle of alpha*A@B + beta*C + diagonal*I and mirrors it. Products
    commute in exact arithmetic; mirroring controls finite-precision asymmetry.
    """
    assert X.ndim in (2, 3) and X.size(-2) > X.size(-1)
    assert gram.shape == (*X.shape[:-2], X.size(-1), X.size(-1))
    assert len(coefficients) > 0 and len(coefficients) % 2 == 0
    q1, q2, scratch = (torch.empty_like(gram) for _ in range(3))
    result = torch.empty_like(X)
    mm = torch.bmm if X.ndim == 3 else torch.mm
    for first in range(0, len(coefficients), 2):
        if first:
            gram_out(X, out=gram)
        a, b, c = coefficients[first]
        sym_product_out(gram, gram, gram, alpha=c, beta=b, diagonal=a, out=q1)
        # R1 = Q1 R Q1. No rectangular update or extra Gram GEMM here.
        sym_product_out(gram, q1, gram, alpha=1., beta=0., diagonal=0., out=scratch)
        sym_product_out(q1, scratch, q1, alpha=1., beta=0., diagonal=0., out=gram)
        a, b, c = coefficients[first + 1]
        sym_product_out(gram, gram, gram, alpha=c, beta=b, diagonal=a, out=q2)
        sym_product_out(q1, q2, q1, alpha=1., beta=0., diagonal=0., out=scratch)
        mm(X, scratch, out=result)
        X, result = result, X
    return X

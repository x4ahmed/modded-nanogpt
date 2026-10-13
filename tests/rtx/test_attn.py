import os, sys
os.environ["PORTABLE"]="1"
sys.path.insert(0, sys.argv[1])
import torch
from torch.nn.attention.flex_attention import flex_attention
from track_1_short.portable.attention import build_window_masks
torch.manual_seed(0)

def reference(q, k, v, doc, window, scale):
    # FA3 varlen causal window semantics: same doc, kv <= q, q - kv <= window. q,k,v: [T,H,D]
    T = q.shape[0]
    i = torch.arange(T)[:, None]; j = torch.arange(T)[None, :]
    allowed = (j <= i) & (i - j <= window) & (doc[:, None] == doc[None, :])
    s = torch.einsum("thd,shd->hts", q.float(), k.float()) * scale
    s = s.masked_fill(~allowed, float("-inf"))
    return torch.einsum("hts,shd->thd", s.softmax(-1), v.float())

T, H = 512, 2
for qk_dim, v_dim in ((64, 128), (64, 64), (128, 128)):
    # documents at irregular boundaries, padded cu_seqlens like the loader's
    bounds = torch.tensor([0, 37, 200, 201, 333, 512, 512, 512], dtype=torch.int32)
    q = torch.randn(T, H, qk_dim); k = torch.randn(T, H, qk_dim); v = torch.randn(T, H, v_dim)
    for window in (128, 256, 384):
        ns, nl, ps = build_window_masks(bounds, T, window, window + 128)
        doc = torch.bucketize(torch.arange(T), bounds[1:6].long(), right=True)
        for mask, w in ((ns, window), (nl, window + 128)):
            y = flex_attention(q.transpose(0,1)[None], k.transpose(0,1)[None], v.transpose(0,1)[None], block_mask=mask, scale=0.12)[0].transpose(0,1)
            ref = reference(q, k, v, doc, w, 0.12)
            err = (y - ref).abs().max().item()
            assert err < 1e-4, (qk_dim, v_dim, window, err)
        # paired: positions doubled, boundaries doubled, same window
        q2 = torch.randn(2*T, H, qk_dim); k2 = torch.randn(2*T, H, qk_dim); v2 = torch.randn(2*T, H, v_dim)
        y = flex_attention(q2.transpose(0,1)[None], k2.transpose(0,1)[None], v2.transpose(0,1)[None], block_mask=ps, scale=0.12)[0].transpose(0,1)
        doc2 = torch.bucketize(torch.arange(2*T), (2*bounds[1:6]).long(), right=True)
        ref = reference(q2, k2, v2, doc2, window, 0.12)
        err = (y - ref).abs().max().item()
        assert err < 1e-4, ("paired", qk_dim, v_dim, window, err)
print("flex masks match FA3 varlen window semantics (incl. unequal QK/V widths, paired)")

import os, sys, tempfile
os.environ["PORTABLE"]="1"; sys.path.insert(0, sys.argv[1])
import numpy as np, torch
_empty = torch.empty
torch.empty = lambda *a, pin_memory=False, **k: _empty(*a, **k)
from track_1_short.portable.data import MicrobatchLoader, validation_batches, microbatch_cu_seqlens_rows
from track_1_short.data import BOS_ID
rng = np.random.default_rng(0)
d = tempfile.mkdtemp()
def write(name, n):
    toks = rng.integers(0, 50000, n).astype(np.uint16)
    pos = 0
    while pos < n:  # documents of random length
        toks[pos] = BOS_ID; pos += int(rng.integers(20, 3000))
    hdr = np.zeros(256, dtype=np.int32); hdr[0] = 20240520; hdr[1] = 1; hdr[2] = n
    with open(os.path.join(d, name), "wb") as f: f.write(hdr.tobytes()); f.write(toks.tobytes())
    return toks
for i in range(2): write(f"fineweb_train_{i:06d}.bin", 3_000_000)
val = write("fineweb_val_000000.bin", 2_200_000)
class Staging:
    def upload(self, *ts): return tuple(t.clone() for t in ts)
mb = 16384
loader = MicrobatchLoader(os.path.join(d, "fineweb_train_*.bin"), mb, Staging())
for step, (K, msl) in enumerate([(8, 896), (16, 2048), (24, 3072), (20, 3072), (24, 3072), (24, 3072)]):
    bs = loader.next_step(K, msl)
    assert len(bs) == K
    for b in bs:
        assert b.inputs.numel() == mb and b.targets.numel() == mb and b.ngram_ids.numel() == 2 * mb
        assert b.inputs[0].item() == BOS_ID, "slice starts at a document start"
        cs = b.cum_seqlens
        assert cs.numel() == microbatch_cu_seqlens_rows(mb) and cs[0] == 0 and (cs[1:] >= cs[:-1]).all() and cs[-1] == mb
        lens = torch.diff(torch.cat([cs, torch.tensor([mb])]).unique_consecutive())
        assert lens.max().item() <= min(msl, 2560), (lens.max().item(), msl)
print("training microbatches OK (shapes, BOS starts, segment caps)")
L = 262144
vb = list(validation_batches(os.path.join(d, "fineweb_val_*.bin"), 8 * L, L, Staging()))
for k, b in enumerate(vb):
    exp = torch.from_numpy(val[k * L:(k + 1) * L].astype(np.int32))
    assert torch.equal(b.inputs, exp) and torch.equal(b.targets, torch.from_numpy(val[k*L+1:(k+1)*L+1].astype(np.int64)))
print(f"validation chunks OK: {len(vb)} chunks equal the record's per-rank layout")

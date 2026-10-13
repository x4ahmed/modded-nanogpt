import os, sys
os.environ["PORTABLE"]="1"; sys.path.insert(0, sys.argv[1])
import torch
from track_1_short.portable.grads import GradAccumulator
from track_1_short.optim.anvil import ParamConfig
a = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16)); b = torch.nn.Parameter(torch.zeros(3)); c = torch.nn.Parameter(torch.zeros(2))
mk = lambda label, optim: ParamConfig(label=label, optim=optim, comms="sharded", adam_betas=None, lr_mul=1, wd_mul=1, lr=1, initial_lr=1, weight_decay=0)
acc = GradAccumulator({a: mk("qk_bank", "anvil"), b: mk("embed", "adam"), c: mk("scalars", "adam")})
for step in (0, 1):
    for _ in range(4):
        a.grad = torch.ones(4, 4, dtype=torch.bfloat16); b.grad = torch.full((3,), 2.0)
        acc.collect()
    acc.release(do_adam=step % 2 == 1)
    assert torch.allclose(a.grad.float(), torch.full((4, 4), 4 / 8))
    if step == 0: assert b.grad is None and c.grad is None
    else: assert torch.allclose(b.grad, torch.full((3,), 16 / 8)) and c.grad is None
print("grad accumulator: per-step 1/8 scaling, two-step Adam window, untouched params skipped")

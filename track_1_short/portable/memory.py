"""GPU memory reporting for the single-GPU port: what each component holds, and the run's peak.

The memory preflight (RTX_PREFLIGHT=1) runs the warmup -- which visits every compiled configuration,
including the largest training microbatch, both softmax kinds and the table updates -- plus one
validation forward at full length, then prints the peak against the device's capacity and exits. A
peak that leaves less than RTX_MIN_HEADROOM_GIB (default 1.5) free fails the preflight.
"""
import os

import torch

GIB = 1024 ** 3
MIN_HEADROOM_GIB = float(os.environ.get("RTX_MIN_HEADROOM_GIB", "1.5"))


def gib(nbytes: int | float) -> str:
    return f"{nbytes / GIB:.2f} GiB"


class MemoryReport:
    def __init__(self, print0, device: torch.device):
        self.print0 = print0
        self.device = device
        self.total = torch.cuda.get_device_properties(device).total_memory
        self.previous = 0

    def mark(self, label: str):
        allocated = torch.cuda.memory_allocated(self.device)
        self.print0(f"MEMORY {label}: allocated {gib(allocated)} (+{gib(allocated - self.previous)}), "
                    f"reserved {gib(torch.cuda.memory_reserved(self.device))}, "
                    f"peak {gib(torch.cuda.max_memory_allocated(self.device))}", console=True)
        self.previous = allocated

    def component(self, label: str, nbytes: int):
        self.print0(f"MEMORY component {label}: {gib(nbytes)}", console=True)

    def verdict(self) -> bool:
        """Peak reserved memory against capacity; True if the headroom is at least MIN_HEADROOM_GIB."""
        free, total = torch.cuda.mem_get_info(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        # Memory outside PyTorch's allocator (CUDA context, cuBLAS/NCCL workspaces) counts against the card.
        outside = (total - free) - torch.cuda.memory_reserved(self.device)
        headroom = total - peak_reserved - outside
        ok = headroom >= MIN_HEADROOM_GIB * GIB
        self.print0(
            f"MEMORY verdict: peak allocated {gib(torch.cuda.max_memory_allocated(self.device))}, peak reserved "
            f"{gib(peak_reserved)}, outside the allocator {gib(outside)}, device {gib(total)} -> headroom "
            f"{gib(headroom)} (need >= {MIN_HEADROOM_GIB:g} GiB): {'PASS' if ok else 'FAIL'}", console=True)
        return ok

r"""Profile the reference wavelet transform: where do time and memory go?

Haar and db2 at s=0.5 on the stage-1 shape [128,16,32,32].
No training, no optimizer steps.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from continuation.transforms.wavelet import wavelet_shrink  # noqa: E402

SHAPE = (128, 16, 32, 32)
S = 0.5


def run(wv, backward):
    dev = torch.device("cuda")
    x = torch.rand(*SHAPE, device=dev, requires_grad=backward)

    def step():
        if backward and x.grad is not None:
            x.grad = None
        y = wavelet_shrink(x, S, wv)
        if backward:
            y.sum().backward()

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(dev)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True, record_shapes=False) as prof:
        for _ in range(5):
            step()
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(dev) / 1024 ** 2
    print("\n===== %s | %s | peak %.0f MB ====="
          % (wv, "fwd+bwd" if backward else "fwd", peak))
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=8))
    return peak


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    for wv in ("haar", "db2"):
        for bwd in (False, True):
            run(wv, bwd)

r"""Cost of wavelet shrinkage (s=0.5) vs Gaussian filtering (sigma=1).

Matched backend, device, dtype and tensor shapes.  The wavelet timing covers the
**complete** operation: mirror extension, two-level analysis, per-band RMS,
soft thresholding, adjoint synthesis and cropping.

Timings are medians over repeated calls after warm-up, with CUDA synchronisation
around each measured region.  Host/device transfers and file I/O are excluded,
and no optimizer step is performed anywhere.

    py scripts/wavelet_benchmark.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from continuation.transforms.gaussian import GaussianSmoothing  # noqa: E402
from continuation.transforms.wavelet import WAVELETS, wavelet_shrink  # noqa: E402

# Representative shapes: the network input and the three ResNet-20 stage
# feature-map shapes at batch size 128.  Benchmarked only -- the training
# architecture is not modified.
SHAPES = {
    "input [128,3,32,32]": (128, 3, 32, 32),
    "stage1 [128,16,32,32]": (128, 16, 32, 32),
    "stage2 [128,32,16,16]": (128, 32, 16, 16),
    "stage3 [128,64,8,8]": (128, 64, 8, 8),
}
S = 0.5
SIGMA = 1.0


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device, repeats=20, warmup=5):
    """Median wall time per call, in milliseconds."""
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def peak_memory_mb(fn, device):
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def gaussian_for(shape, sigma_max):
    """A Gaussian whose fixed support fits the given spatial size."""
    g = GaussianSmoothing(sigma_max=sigma_max, truncate=4.0)
    if g.radius >= min(shape[-2], shape[-1]):
        return None
    return g


def bench_device(device, dtype=torch.float32, repeats=20):
    out = {}
    for label, shape in SHAPES.items():
        x = torch.rand(*shape, device=device, dtype=dtype)
        entry = {"shape": list(shape), "forward_ms": {}, "forward_backward_ms": {},
                 "peak_mem_mb": {}}

        for wv in WAVELETS:
            entry["forward_ms"]["wavelet_%s" % wv] = timed(
                lambda w=wv: wavelet_shrink(x, S, w), device, repeats)

        # sigma_max=3 is the Experiment-0 configuration (radius 12); sigma_max=1
        # is the smallest support that still covers sigma=1 (radius 4) and is the
        # only one that fits the 8x8 stage-3 maps.
        for tag, smax in (("gaussian_sigma_max3", 3.0), ("gaussian_sigma_max1", 1.0)):
            g = gaussian_for(shape, smax)
            entry["forward_ms"][tag] = (timed(lambda gg=g: gg(x, SIGMA), device, repeats)
                                        if g is not None else None)

        if device.type == "cuda":
            xg = x.clone().requires_grad_(True)

            def fb_wav(w):
                if xg.grad is not None:
                    xg.grad = None
                wavelet_shrink(xg, S, w).sum().backward()

            def fb_gauss(g):
                if xg.grad is not None:
                    xg.grad = None
                g(xg, SIGMA).sum().backward()

            for wv in WAVELETS:
                entry["forward_backward_ms"]["wavelet_%s" % wv] = timed(
                    lambda w=wv: fb_wav(w), device, repeats)
                entry["peak_mem_mb"]["wavelet_%s" % wv] = peak_memory_mb(
                    lambda w=wv: fb_wav(w), device)
            for tag, smax in (("gaussian_sigma_max3", 3.0), ("gaussian_sigma_max1", 1.0)):
                g = gaussian_for(shape, smax)
                entry["forward_backward_ms"][tag] = (
                    timed(lambda gg=g: fb_gauss(gg), device, repeats) if g else None)
                entry["peak_mem_mb"][tag] = (
                    peak_memory_mb(lambda gg=g: fb_gauss(gg), device) if g else None)
        out[label] = entry
    return out


def main():
    report = {
        "s": S, "sigma": SIGMA, "dtype": "float32",
        "wavelet_scope": ("complete operation: mirror extension, 2-level analysis, "
                          "per-band RMS, soft threshold, adjoint synthesis, crop"),
        "method": ("median over repeated calls after warm-up; CUDA-synchronised; "
                   "excludes host/device transfers and file I/O; no optimizer steps"),
        "torch": torch.__version__,
    }

    cpu = torch.device("cpu")
    torch.set_num_threads(6)
    single = torch.rand(1, 3, 32, 32, device=cpu)
    report["cpu_single_image_ms"] = {}
    for wv in WAVELETS:
        report["cpu_single_image_ms"]["wavelet_%s" % wv] = timed(
            lambda w=wv: wavelet_shrink(single, S, w), cpu, repeats=50, warmup=10)
    for tag, smax in (("gaussian_sigma_max3", 3.0), ("gaussian_sigma_max1", 1.0)):
        g = GaussianSmoothing(sigma_max=smax, truncate=4.0)
        report["cpu_single_image_ms"][tag] = timed(
            lambda gg=g: gg(single, SIGMA), cpu, repeats=50, warmup=10)

    print("=== CPU, single image [1,3,32,32], forward (ms) ===")
    base = report["cpu_single_image_ms"]["gaussian_sigma_max1"]
    for k, v in report["cpu_single_image_ms"].items():
        print("  %-24s %8.3f  (%.1fx gaussian_sigma_max1)" % (k, v, v / base))

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        report["gpu"] = {"name": torch.cuda.get_device_name(0),
                         "free_mb": round(free / 1024 ** 2),
                         "total_mb": round(total / 1024 ** 2)}
        print("\n=== GPU (%s), batch shapes ===" % report["gpu"]["name"])
        report["cuda"] = bench_device(torch.device("cuda"))
        for label, e in report["cuda"].items():
            print("\n  %s" % label)
            for k in sorted(e["forward_ms"]):
                f = e["forward_ms"][k]
                fb = e["forward_backward_ms"].get(k)
                mem = e["peak_mem_mb"].get(k)
                print("    %-24s fwd %8s  fwd+bwd %8s  peak %8s MB"
                      % (k,
                         "n/a" if f is None else "%.3f" % f,
                         "n/a" if fb is None else "%.3f" % fb,
                         "n/a" if mem is None else "%.1f" % mem))
    else:
        report["gpu"] = None
        print("\nGPU unavailable; CPU results only.")

    out = ROOT / "results" / "wavelet_previews" / "wavelet_timings.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()

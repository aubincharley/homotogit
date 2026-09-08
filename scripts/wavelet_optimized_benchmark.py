r"""Before/after benchmark for the optimized wavelet transform.

Same device, dtype, shapes, warm-up and CUDA synchronisation as
``wavelet_benchmark.py``.  Graphs are never retained across repetitions and
memory statistics are reset before every measurement.  No optimizer steps.

    py scripts/wavelet_optimized_benchmark.py
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

SHAPES = {
    "input [128,3,32,32]": (128, 3, 32, 32),
    "stage1 [128,16,32,32]": (128, 16, 32, 32),
    "stage2 [128,32,16,16]": (128, 32, 16, 16),
    "stage3 [128,64,8,8]": (128, 64, 8, 8),
}
S = 0.5


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def measure(fn, dev, repeats=15, warmup=5):
    """Median ms and peak MB, with a clean memory baseline."""
    for _ in range(warmup):
        fn()
    sync(dev)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    samples = []
    for _ in range(repeats):
        sync(dev)
        t0 = time.perf_counter()
        fn()
        sync(dev)
        samples.append((time.perf_counter() - t0) * 1000.0)
    peak = torch.cuda.max_memory_allocated(dev) / 1024 ** 2 if dev.type == "cuda" else None
    return statistics.median(samples), peak


def variants(x, s=S):
    """(label, callable) pairs.  ``x`` may or may not require grad."""
    out = []
    for wv in WAVELETS:
        out.append(("ref/%s" % wv,
                    lambda w=wv: wavelet_shrink(x, s, w, impl="reference")))
        out.append(("fast/%s" % wv,
                    lambda w=wv: wavelet_shrink(x, s, w, impl="fast")))
        out.append(("fast+ckpt/%s" % wv,
                    lambda w=wv: wavelet_shrink(x, s, w, impl="fast", checkpoint=True)))
    return out


def gaussian_for(shape, sigma_max=1.0):
    g = GaussianSmoothing(sigma_max=sigma_max, truncate=4.0)
    return None if g.radius >= min(shape[-2], shape[-1]) else g


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"device": str(dev), "s": S, "dtype": "float32",
              "torch": torch.__version__, "shapes": {}}
    if dev.type == "cuda":
        report["gpu"] = {"name": torch.cuda.get_device_name(0),
                         "total_mb": round(torch.cuda.get_device_properties(0)
                                           .total_memory / 1024 ** 2)}
    print("device: %s" % (report.get("gpu", {}).get("name", dev)))

    for label, shape in SHAPES.items():
        entry = {"forward": {}, "forward_backward": {}}
        x = torch.rand(*shape, device=dev)
        for name, fn in variants(x):
            if "ckpt" in name:
                continue                      # checkpointing is a backward-only win
            ms, mb = measure(fn, dev)
            entry["forward"][name] = {"ms": ms, "peak_mb": mb}

        xg = torch.rand(*shape, device=dev, requires_grad=True)

        def bwd(fn):
            if xg.grad is not None:
                xg.grad = None
            fn().sum().backward()             # graph freed each iteration

        for name, fn in variants(xg):
            ms, mb = measure(lambda f=fn: bwd(f), dev)
            entry["forward_backward"][name] = {"ms": ms, "peak_mb": mb}

        g = gaussian_for(shape)
        if g is not None:
            ms, mb = measure(lambda: g(x, 1.0), dev)
            entry["forward"]["gaussian_sigma1"] = {"ms": ms, "peak_mb": mb,
                                                   "kernel": g.kernel_size,
                                                   "radius": g.radius}
            ms, mb = measure(lambda: bwd(lambda: g(xg, 1.0)), dev)
            entry["forward_backward"]["gaussian_sigma1"] = {
                "ms": ms, "peak_mb": mb, "kernel": g.kernel_size, "radius": g.radius}
        report["shapes"][label] = entry

        print("\n%s" % label)
        for k in sorted(entry["forward"]):
            f = entry["forward"][k]
            b = entry["forward_backward"].get(k, {})
            print("  %-18s fwd %8.3f ms (%7s MB) | fwd+bwd %8s ms (%7s MB)"
                  % (k, f["ms"], "%.0f" % f["peak_mb"] if f["peak_mb"] else "-",
                     "%.3f" % b["ms"] if b else "-",
                     "%.0f" % b["peak_mb"] if b.get("peak_mb") else "-"))
        for k in sorted(entry["forward_backward"]):
            if k not in entry["forward"]:
                b = entry["forward_backward"][k]
                print("  %-18s fwd        -           | fwd+bwd %8.3f ms (%7s MB)"
                      % (k, b["ms"], "%.0f" % b["peak_mb"] if b["peak_mb"] else "-"))

    # s-sweep on one representative shape, including the identity bypass
    shape = SHAPES["stage1 [128,16,32,32]"]
    x = torch.rand(*shape, device=dev, requires_grad=True)
    sw = {}
    print("\n=== s sweep, %s, fast path, db2 ===" % str(shape))
    for s in (0.0, 0.5, 0.9, 1.0):
        for bypass in ((False, True) if s == 1.0 else (False,)):
            def step(ss=s, bp=bypass):
                if x.grad is not None:
                    x.grad = None
                wavelet_shrink(x, ss, "db2", impl="fast",
                               bypass_identity=bp).sum().backward()
            ms, mb = measure(step, dev)
            tag = "s=%g%s" % (s, " (identity bypass)" if bypass else "")
            sw[tag] = {"ms": ms, "peak_mb": mb}
            print("  %-24s fwd+bwd %8.3f ms  peak %7s MB"
                  % (tag, ms, "%.0f" % mb if mb else "-"))
    report["s_sweep_stage1_db2_fwd_bwd"] = sw

    out = ROOT / "results" / "wavelet_previews" / "wavelet_optimized_timings.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()

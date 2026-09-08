r"""Is the wavelet transform affordable inside ResNet-20?  Cost assessment only.

A **temporary** benchmark wrapper attaches a forward hook to every spatial 3x3
convolution and applies the transform to that convolution's output, i.e. before
the following GroupNorm/ReLU.  Shortcuts are untouched.  The model definition is
not modified and the hooks are removed afterwards -- this is a placement for
measuring cost, not a training design.

Short forward / forward+backward benchmarks only.  Model parameters require
gradients; **no optimizer is constructed and no optimizer step is taken.**

    py scripts/wavelet_network_feasibility.py
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
import torch.nn as nn  # noqa: E402

from continuation.config import ModelConfig  # noqa: E402
from continuation.models import build_model  # noqa: E402
from continuation.transforms.gaussian import GaussianSmoothing  # noqa: E402
from continuation.transforms.wavelet import wavelet_shrink  # noqa: E402

EFFECTIVE_BATCH = 128
TOTAL_UPDATES = 14040
S = 0.5
CANDIDATES = [128, 64, 32, 16, 8, 4, 2]


def attach(model, fn):
    """Hook every 3x3 conv so its output passes through ``fn``."""
    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            handles.append(m.register_forward_hook(lambda _m, _i, out, f=fn: f(out)))
    return handles


def make_variant(name):
    if name == "baseline":
        return None
    if name == "gaussian_sigma1":
        # radius 4 (9-tap): the smallest support covering sigma=1 and the only
        # one that fits the 8x8 stage-3 maps.
        g = GaussianSmoothing(sigma_max=1.0, truncate=4.0)
        return lambda t: g(t, 1.0)
    if name.startswith("wavelet_"):
        wv = name.split("_", 1)[1]
        return lambda t: wavelet_shrink(t, S, wv, impl="fast")
    raise KeyError(name)


def bench(variant, batch, dev, repeats=8, warmup=3):
    model = build_model(ModelConfig(), 10, seed=0).to(dev).train()
    for p in model.parameters():
        p.requires_grad_(True)
    fn = make_variant(variant)
    handles = attach(model, fn) if fn else []
    x = torch.rand(batch, 3, 32, 32, device=dev)
    y = torch.randint(0, 10, (batch,), device=dev)
    lossf = nn.CrossEntropyLoss()

    def fwd():
        with torch.no_grad():
            model(x)

    def fwd_bwd():
        model.zero_grad(set_to_none=True)
        lossf(model(x), y).backward()

    try:
        for _ in range(warmup):
            fwd_bwd()
        torch.cuda.synchronize(dev) if dev.type == "cuda" else None
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
        out = {}
        for label, f in (("forward", fwd), ("forward_backward", fwd_bwd)):
            s = []
            for _ in range(repeats):
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                f()
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                s.append((time.perf_counter() - t0) * 1000.0)
            out[label + "_ms"] = statistics.median(s)
        out["peak_mb"] = (torch.cuda.max_memory_allocated(dev) / 1024 ** 2
                          if dev.type == "cuda" else None)
        out["n_hooks"] = len(handles)
        return out
    finally:
        for h in handles:
            h.remove()
        del model
        if dev.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def _probe_subprocess(variant, batch):
    """Run one (variant, batch) probe in a fresh process.

    A CUDA OOM can leave the allocator/context in a state that later calls
    cannot recover from, so each probe is isolated; the parent only sees a
    clean success/failure.
    """
    import subprocess

    code = (
        "import json,sys;sys.path.insert(0,%r);"
        "from scripts.wavelet_network_feasibility import bench;"
        "import torch;"
        "d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');"
        "r=bench(%r,%d,d);print('RESULT'+json.dumps(r))" % (str(ROOT), variant, batch)
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[len("RESULT"):])
    return None


VRAM_CEILING = 0.85          # fraction of total VRAM a probe may peak at


def largest_fitting(variant, dev, total_vram_mb=None):
    """Largest microbatch that genuinely fits in VRAM.

    On Windows/WDDM a too-large allocation does **not** raise OOM: the driver
    silently oversubscribes into host memory, so the probe "succeeds" while
    thrashing over PCIe (we measured a peak of 8712 MB on a 4096 MB card and
    67 s per step).  A hard OOM is therefore not a reliable fit test; we also
    require the measured peak to stay under a fraction of physical VRAM.
    """
    for b in CANDIDATES:
        r = _probe_subprocess(variant, b)
        if r is None:
            print("    %s: microbatch %d raised OOM" % (variant, b))
            continue
        peak = r.get("peak_mb")
        if total_vram_mb and peak and peak > VRAM_CEILING * total_vram_mb:
            print("    %s: microbatch %d peaked at %.0f MB > %.0f%% of %d MB VRAM "
                  "(host-memory spill), rejecting"
                  % (variant, b, peak, VRAM_CEILING * 100, total_vram_mb))
            continue
        return b, r
    return None, None


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"device": str(dev), "effective_batch": EFFECTIVE_BATCH,
              "total_updates": TOTAL_UPDATES, "s": S, "torch": torch.__version__,
              "placement": "forward hook on every 3x3 conv output, before GN/ReLU; "
                           "shortcuts untouched; temporary, not a training design",
              "variants": {}}
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        report["gpu"] = {"name": props.name,
                         "total_vram_mb": round(props.total_memory / 1024 ** 2)}
        print("GPU: %s, %d MB VRAM" % (props.name, report["gpu"]["total_vram_mb"]))

    for variant in ("baseline", "gaussian_sigma1", "wavelet_db2", "wavelet_haar"):
        b, r = largest_fitting(variant, dev,
                               (report.get("gpu") or {}).get("total_vram_mb"))
        if b is None:
            report["variants"][variant] = {"fits": False}
            print("%-16s does not fit at any tested microbatch" % variant)
            continue
        accum = EFFECTIVE_BATCH // b
        per_eff = r["forward_backward_ms"] * accum
        entry = {
            "microbatch": b, "accumulation_steps": accum,
            "n_hooks": r["n_hooks"],
            "forward_ms": r["forward_ms"],
            "forward_backward_ms": r["forward_backward_ms"],
            "peak_mb": r["peak_mb"],
            "ms_per_effective_batch_128": per_eff,
            "compute_only_hours_14040_updates": per_eff * TOTAL_UPDATES / 3.6e6,
        }
        report["variants"][variant] = entry
        print("%-16s microbatch %3d x%-2d accum | fwd %7.2f ms | fwd+bwd %8.2f ms | "
              "peak %6.0f MB | %8.1f ms/eff-batch | %6.2f h for %d updates"
              % (variant, b, accum, r["forward_ms"], r["forward_backward_ms"],
                 r["peak_mb"] or 0, per_eff,
                 entry["compute_only_hours_14040_updates"], TOTAL_UPDATES))

    report["estimate_excludes"] = ("data loading, evaluation/validation, optimizer "
                                   "step and its state, checkpoint I/O")
    out = ROOT / "results" / "wavelet_previews" / "wavelet_network_feasibility.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()

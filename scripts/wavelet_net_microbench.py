r"""One small, time-bounded in-network cost benchmark.  No training, no optimizer.

Three variants at the same 19 insertion points (forward hook on every 3x3 conv
output, before the following GroupNorm/ReLU; shortcuts untouched):
plain ResNet-20, Gaussian ``sigma=1``, and fused ``db2`` at ``s=0.5``.

Fixed microbatch 4, ``[4,3,32,32]`` float32, synthetic inputs and labels.  No
batch-size search, gradient accumulation, checkpointing or compilation.

The parent runs the probe as an unbuffered subprocess under a hard 240 s total
timeout (startup included).  Each completed variant is emitted as a JSON line
immediately, so a timeout or OOM still yields partial results; the probe is then
terminated and never retried.

    py scripts/wavelet_net_microbench.py            # parent (enforces the limit)
    py -u scripts/wavelet_net_microbench.py --child # probe only
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MICROBATCH = 4
SHAPE = (MICROBATCH, 3, 32, 32)
S = 0.5
SIGMA = 1.0
MEM_FRACTION = 0.60
TOTAL_TIMEOUT_S = 240
WARMUP, TIMED = 2, 3
VARIANTS = ("plain", "gaussian_sigma1", "wavelet_db2_fused")


# --------------------------------------------------------------------- child

def _child() -> int:
    import torch
    import torch.nn as nn

    from continuation.config import ModelConfig
    from continuation.models import build_model
    from continuation.transforms.gaussian import GaussianSmoothing
    from continuation.transforms.wavelet import wavelet_shrink

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        # Cap the PyTorch allocator before any model/tensor allocation.  This
        # bounds the allocator only, not total driver memory, which is why the
        # parent's wall-clock timeout is still required.
        torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
        props = torch.cuda.get_device_properties(0)
        meta = {"device": props.name,
                "total_vram_mib": round(props.total_memory / 1024 ** 2),
                "allocator_fraction": MEM_FRACTION}
    else:
        meta = {"device": "cpu", "total_vram_mib": None,
                "allocator_fraction": None}
    print("META " + json.dumps(meta), flush=True)

    def hook_fn(name):
        if name == "plain":
            return None
        if name == "gaussian_sigma1":
            g = GaussianSmoothing(sigma_max=1.0, truncate=4.0)   # radius 4, 9 taps
            return lambda t: g(t, SIGMA)
        return lambda t: wavelet_shrink(t, S, "db2", impl="fast")

    lossf = nn.CrossEntropyLoss()
    for variant in VARIANTS:
        model = build_model(ModelConfig(), 10, seed=0).to(dev).train()
        for p in model.parameters():
            p.requires_grad_(True)
        fn = hook_fn(variant)
        handles = []
        if fn is not None:
            for m in model.modules():
                if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                    handles.append(m.register_forward_hook(
                        lambda _m, _i, out, f=fn: f(out)))
        x = torch.rand(*SHAPE, device=dev)
        y = torch.randint(0, 10, (MICROBATCH,), device=dev)

        def step():
            model.zero_grad(set_to_none=True)          # clear grads between passes
            loss = lossf(model(x), y)
            loss.backward()                            # graph released here
            del loss

        try:
            for i in range(WARMUP):
                print("  %s warmup %d/%d ..." % (variant, i + 1, WARMUP), flush=True)
                step()
                print("  %s warmup %d/%d done" % (variant, i + 1, WARMUP), flush=True)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)   # per-variant reset
            times = []
            for i in range(TIMED):
                print("  %s pass %d/%d ..." % (variant, i + 1, TIMED), flush=True)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                step()
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                dt = (time.perf_counter() - t0) * 1000.0
                times.append(dt)
                print("  %s pass %d/%d done %.1f ms" % (variant, i + 1, TIMED, dt),
                      flush=True)
            rec = {"variant": variant, "n_hooks": len(handles),
                   "median_fwd_bwd_ms": statistics.median(times),
                   "times_ms": times, "status": "ok"}
            if dev.type == "cuda":
                rec["peak_allocated_mib"] = torch.cuda.max_memory_allocated(dev) / 1024 ** 2
                rec["peak_reserved_mib"] = torch.cuda.max_memory_reserved(dev) / 1024 ** 2
        except Exception as exc:                          # OOM or anything else
            rec = {"variant": variant, "n_hooks": len(handles),
                   "status": "failed", "error": type(exc).__name__, "detail": str(exc)[:300]}
        print("RESULT " + json.dumps(rec), flush=True)

        for h in handles:
            h.remove()
        del model, x, y
        if dev.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return 0


# -------------------------------------------------------------------- parent

def _parent() -> int:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--child"]
    print("running probe with a hard %d s budget: %s" % (TOTAL_TIMEOUT_S, " ".join(cmd)),
          flush=True)
    t0 = time.perf_counter()
    results, meta, timed_out = [], {}, False
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(line, flush=True)
            if line.startswith("RESULT "):
                results.append(json.loads(line[len("RESULT "):]))
            elif line.startswith("META "):
                meta = json.loads(line[len("META "):])
            if time.perf_counter() - t0 > TOTAL_TIMEOUT_S:
                timed_out = True
                break
        else:
            proc.wait(timeout=max(1, TOTAL_TIMEOUT_S - (time.perf_counter() - t0)))
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        if proc.poll() is None:
            timed_out = True
            proc.kill()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True)
    elapsed = time.perf_counter() - t0

    missing = [v for v in VARIANTS if not any(r["variant"] == v for r in results)]
    report = {"microbatch": MICROBATCH, "shape": list(SHAPE), "dtype": "float32",
              "s": S, "sigma": SIGMA, "timeout_s": TOTAL_TIMEOUT_S,
              "elapsed_s": round(elapsed, 1), "timed_out": timed_out,
              "environment": meta, "results": results, "not_measured": missing,
              "note": ("stalled or failed variants are reported as-is; no training-time "
                       "extrapolation is made from them")}

    total = meta.get("total_vram_mib")
    print("\n%-20s %10s %14s %14s %8s" % ("variant", "fwd+bwd ms", "peak alloc MiB",
                                          "peak resv MiB", "status"))
    for v in VARIANTS:
        r = next((x for x in results if x["variant"] == v), None)
        if r is None:
            print("%-20s %10s %14s %14s %8s" % (v, "-", "-", "-", "not run"))
            continue
        alloc, resv = r.get("peak_allocated_mib"), r.get("peak_reserved_mib")
        flag = ""
        if total and alloc and alloc > total:
            flag = "  <-- exceeds physical VRAM (host-memory spill)"
        elif total and alloc and alloc > 0.85 * total:
            flag = "  <-- near VRAM limit"
        print("%-20s %10s %14s %14s %8s%s"
              % (v,
                 "%.1f" % r["median_fwd_bwd_ms"] if r.get("median_fwd_bwd_ms") else "-",
                 "%.0f" % alloc if alloc else "-",
                 "%.0f" % resv if resv else "-",
                 r["status"], flag))
    if timed_out:
        print("\nTIMED OUT after %.0f s; probe terminated. Partial results only." % elapsed)

    out = ROOT / "results" / "wavelet_previews" / "wavelet_net_microbench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    sys.exit(_child() if ap.parse_args().child else _parent())

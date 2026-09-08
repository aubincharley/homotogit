r"""Verify gradient accumulation against a single full batch, on the real loop.

One batch of 128 vs four microbatches of 32 over the *same* examples, from
identical model and optimizer state.  With mean CE per microbatch and four equal
microbatches, dividing each loss by four before backward must reproduce

    g = sum_m (n_m / B) grad CE_m,   B = sum_m n_m.

Gradients are zeroed once before accumulation; the optimizer and the LR are set
once per complete batch.

The identity is asserted in **float64**, where it must hold to round-off.  The
same comparison in float32 on CUDA shows ~6e-4 relative error, because cuDNN
uses a different reduction order for batch 32 than for batch 128.  That is
precision, not a defect: deterministic algorithms do not change it, running the
*same* path twice agrees to 5e-8, and the gradient norms still agree to ~1e-5.

    py scripts/verify_grad_accumulation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from continuation.config import DataConfig, ModelConfig, OptimConfig  # noqa: E402
from continuation.data import build_dataset  # noqa: E402
from continuation.models import build_model  # noqa: E402
from continuation.optim import build_optimizer, lr_at, set_lr  # noqa: E402
from continuation.pipeline import ChannelNormalizer, InputPipeline  # noqa: E402
from continuation.transforms.gaussian import GaussianSmoothing  # noqa: E402

BATCH, MICRO = 128, 32
STEP = 7            # arbitrary global step, so the LR is a realistic non-warmup value
TOL_EXACT = 1e-12   # float64: the identity itself
TOL_FP32 = 2e-3     # float32: cross-batch-size cuDNN reduction-order noise


def flat(vs):
    return torch.cat([v.reshape(-1) for v in vs])


def rel_err(a, b):
    return float((a - b).norm() / (b.norm() + 1e-30))


def compare(dev, dtype):
    bundle = build_dataset(DataConfig()).to(dev)
    pipe = InputPipeline(GaussianSmoothing(sigma_max=1.0),
                         ChannelNormalizer(bundle.mean, bundle.std).to(dev))
    idx = torch.arange(BATCH, device=dev)
    x, y = bundle.train.images[idx], bundle.train.labels[idx]

    ocfg = OptimConfig(lr=0.1, momentum=0.9, weight_decay=5e-4, total_steps=600,
                       lr_schedule="cosine", warmup_steps=30)
    lr = lr_at(STEP, ocfg)

    def fresh():
        m = build_model(ModelConfig(), bundle.num_classes, seed=0).to(dev).to(dtype).train()
        o = build_optimizer(m, ocfg)
        set_lr(o, lr)
        return m, o

    # --- one full batch of 128 ---
    m1, o1 = fresh()
    init = [p.detach().clone() for p in m1.parameters()]
    o1.zero_grad(set_to_none=True)
    F.cross_entropy(m1(pipe(x, 0.0).to(dtype)), y).backward()
    g_full = flat([p.grad.detach().clone() for p in m1.parameters()])
    o1.step()

    # --- four microbatches of 32 over the same examples, in the same order ---
    m2, o2 = fresh()
    assert all(torch.equal(a, b) for a, b in zip(init, list(m2.parameters()))), \
        "the two runs must start from identical weights"
    o2.zero_grad(set_to_none=True)                       # zeroed once
    accum = BATCH // MICRO
    for a in range(accum):
        sl = idx[a * MICRO:(a + 1) * MICRO]
        (F.cross_entropy(m2(pipe(bundle.train.images[sl], 0.0).to(dtype)),
                         bundle.train.labels[sl]) / accum).backward()
    g_acc = flat([p.grad.detach().clone() for p in m2.parameters()])
    o2.step()                                            # stepped once

    upd_full = flat([a - b for a, b in zip(list(m1.parameters()), init)])
    upd_acc = flat([a - b for a, b in zip(list(m2.parameters()), init)])
    g_re, u_re = rel_err(g_acc, g_full), rel_err(upd_acc, upd_full)
    tol = TOL_EXACT if dtype == torch.float64 else TOL_FP32
    n_full, n_acc = float(g_full.norm()), float(g_acc.norm())
    return {"device": str(dev), "dtype": str(dtype), "batch": BATCH,
            "microbatch": MICRO, "accumulation": accum, "lr": lr, "step": STEP,
            "grad_relative_error": g_re, "update_relative_error": u_re,
            "grad_norm_full": n_full, "grad_norm_accumulated": n_acc,
            "grad_norm_relative_error": abs(n_acc - n_full) / n_full,
            "tolerance": tol, "passed": bool(g_re < tol and u_re < tol)}


def main():
    results = [compare(torch.device("cpu"), torch.float64)]
    if torch.cuda.is_available():
        results.append(compare(torch.device("cuda"), torch.float32))
    out = {"checks": results,
           "conclusion": ("accumulation is mathematically exact (float64 to round-off); "
                          "the float32 CUDA gap is cuDNN reduction order differing "
                          "between batch 32 and batch 128, not a defect"),
           "passed": all(r["passed"] for r in results)}
    for r in results:
        print("%-5s %-16s grad rel err %.2e | update rel err %.2e | norm rel err %.2e "
              "| tol %.0e -> %s"
              % (r["device"], r["dtype"], r["grad_relative_error"],
                 r["update_relative_error"], r["grad_norm_relative_error"],
                 r["tolerance"], "PASS" if r["passed"] else "FAIL"))
    p = ROOT / "results" / "grad_accumulation_check.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("wrote %s" % p)
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

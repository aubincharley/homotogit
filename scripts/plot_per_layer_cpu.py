"""Figure for the per-layer sigma CPU pilot.

Follows the plotting rules recorded in docs/HANDOVER.md:

* the **current path** (what the weights and BN buffers were trained under) is
  the primary curve; the target path is a thin dashed diagnostic, because while
  filters are active it measures a premature configuration change including
  BatchNorm mismatch and is not predictor quality;
* axes are clipped to the primary curves, so a BN-mismatch spike cannot flatten
  everything else;
* every line style appears in the legend;
* the rendered PNG is read back rather than trusting the script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR = {"plain": "#444444", "rho1": "#1f77b4",
         "rho0.5": "#2ca02c", "rho2": "#d62728", "adaptive": "#9467bd"}
LABEL = {"plain": "plain (no filter)",
         "rho1": r"$\rho$=1  uniform $\sigma$ (current)",
         "rho0.5": r"$\rho$=0.5  c=(1, ½, ¼) constant physical scale",
         "rho2": r"$\rho$=2  c=(¼, ½, 1) blur deep",
         "adaptive": "adaptive predictor-corrector"}


def main(out_dir="results/per_layer_cpu"):
    out = Path(out_dir)
    summaries = json.loads((out / "summaries.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    lo, hi = 1e9, -1e9
    for s in summaries:
        arm = s["arm"]
        m = json.loads((out / arm / "metrics.json").read_text())
        ep = [r["epoch"] for r in m]
        cur = [r["test_acc_current"] for r in m]
        tgt = [r["test_acc_target"] for r in m]
        ce = [r["test_ce_current"] for r in m]
        c = COLOR.get(arm, "#888888")
        axes[0].plot(ep, cur, "-o", ms=3, color=c, label=LABEL.get(arm, arm))
        axes[0].plot(ep, tgt, "--", lw=0.8, alpha=0.55, color=c)
        axes[1].plot(ep, ce, "-o", ms=3, color=c, label=LABEL.get(arm, arm))
        lo, hi = min(lo, min(cur)), max(hi, max(cur))

    # mark where the filters switch off
    bypass = None
    m0 = json.loads((out / summaries[-1]["arm"] / "metrics.json").read_text())
    for r in m0:
        if r["row"] is not None and not r["filters_active"] and bypass is None:
            bypass = r["epoch"]
    for ax in axes:
        if bypass is not None:
            ax.axvline(bypass - 0.5, color="#999999", ls=":", lw=1.2,
                       label="filters off (target objective)")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    pad = 0.05 * max(hi - lo, 1e-3)
    axes[0].set_ylim(lo - pad, hi + pad)          # clip to the primary curves
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("Test accuracy — solid: current path, dashed: target path")
    axes[1].set_ylabel("test cross-entropy (current path)")
    axes[1].set_title("Test CE")
    axes[0].legend(fontsize=7.5, loc="lower right")

    n = summaries[0]
    fig.suptitle("Per-layer $\\sigma$ profiles, ResNet-20 BN, CPU pilot — "
                 "%d train / %d test, %d epochs, seed %d (1 seed, exploratory)"
                 % (n["n_train"], n["n_test"], n["epochs"], n["seed"]),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = out / "per_layer_cpu.png"
    fig.savefig(png, dpi=150)
    print("wrote", png, png.stat().st_size, "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:])

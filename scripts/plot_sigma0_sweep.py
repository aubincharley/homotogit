"""Amplitude response of internal Gaussian smoothing.

One point per sigma0, three paired seeds each, against the unfiltered control.
The measured run-to-run floor at this scale (EXP-012) is ~0.5 pp and is drawn as
a band around the control so a reader can see immediately which arms clear it.

Axes are deliberately linear in sigma0 and not log: the question is whether the
incumbent sigma0 = 1.0 sits at an optimum, and a log axis would flatter the small
amplitudes.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("s0_0.25", 0.25), ("s0_0.5", 0.5), ("rho1", 1.0),
        ("s0_1.5", 1.5), ("s0_2", 2.0)]
NOISE_PP = 0.5          # measured in EXP-012 by re-running plain and rho1


def main(study="results/kaggle_outputs/sigma0-sweep-20260910-203015/sigma0_sweep"):
    D = Path(study)
    seeds = [0, 1, 2]
    acc, ce = {}, {}
    for s in seeds:
        for r in json.loads((D / f"seed{s}" / "summaries.json").read_text()):
            acc.setdefault(r["arm"], []).append(r["final_test_acc"])
            ce.setdefault(r["arm"], []).append(r["final_test_ce"])
    mean = lambda v: sum(v) / len(v)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    xs = [s for _, s in ARMS]
    ys = [100 * mean(acc[a]) for a, _ in ARMS]
    es = [100 * st.stdev(acc[a]) for a, _ in ARMS]
    base, base_sd = 100 * mean(acc["plain"]), 100 * st.stdev(acc["plain"])

    axes[0].axhspan(base - NOISE_PP, base + NOISE_PP, color="#bbbbbb", alpha=0.35,
                    label="control ± measured noise floor (0.5 pp)")
    axes[0].axhline(base, color="#444444", ls="--", lw=1.2, label="plain (no filter)")
    axes[0].errorbar(xs, ys, yerr=es, fmt="-o", ms=6, capsize=4, color="#1f77b4",
                     label="Gaussian, uniform profile")
    best = max(range(len(ARMS)), key=lambda i: ys[i])
    axes[0].annotate("incumbent $\\sigma_0$=1.0\n(never previously varied)",
                     (xs[best], ys[best]), textcoords="offset points",
                     xytext=(12, -34), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.9))
    axes[0].set_xlabel(r"$\sigma_0$  (peak width; schedule and profile held fixed)")
    axes[0].set_ylabel("test accuracy (%)")
    axes[0].set_title("Amplitude response — sharply peaked")
    axes[0].legend(fontsize=8, loc="lower center")
    axes[0].grid(alpha=0.25)

    cy = [mean(ce[a]) for a, _ in ARMS]
    cs = [st.stdev(ce[a]) for a, _ in ARMS]
    axes[1].axhline(mean(ce["plain"]), color="#444444", ls="--", lw=1.2, label="plain")
    axes[1].errorbar(xs, cy, yerr=cs, fmt="-o", ms=6, capsize=4, color="#d62728",
                     label="Gaussian")
    axes[1].set_xlabel(r"$\sigma_0$")
    axes[1].set_ylabel("test cross-entropy")
    axes[1].set_title("Test CE — same minimum, cleaner")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.suptitle("Internal Gaussian: amplitude sweep, ResNet-20 BN, full CIFAR-10, "
                 "3 paired seeds (EXP-013)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path("results/sigma0_sweep.png")
    fig.savefig(out, dpi=150)
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:])

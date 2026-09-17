"""Curvature summary figures from the trace and eigenvalue tables (no new evaluation).

    py -m geometry_final.curvature_figs
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .figures import CAPTIONS, COLOR, FIG, LONG, M, TAB, plt, save

QTY = [("lambda_top1_ordinary", r"$\lambda_{\max}(H)$"), ("trace_ordinary", r"$\mathrm{tr}(H)$"),
       ("lambda_top1_relative", r"$\lambda_{\max}(A^\top HA)$"), ("trace_relative", r"$\mathrm{tr}(A^\top HA)$"),
       ("trace_covariance", r"$\mathrm{tr}(HC)=\mathbb{E}\,d^\top Hd$")]


def ratios(name):
    pr = pd.read_csv(TAB / "T_trace_paired_vs_plain.csv")
    fig, axes = plt.subplots(2, len(QTY), figsize=(7.2, 3.9), sharey=True, layout="constrained")
    for r_, probe in enumerate(("train_probe", "test_probe")):
        for c_, (q, lab) in enumerate(QTY):
            ax = axes[r_, c_]
            ax.axhline(1, color=".35", lw=.7)
            for i, m in enumerate(M[1:]):
                g = pr[(pr.probe == probe) & (pr.method == m) & (pr.quantity == q)].sort_values("seed")
                x = i + np.linspace(-.18, .18, len(g))
                yerr = g.ratio_mc_sem.to_numpy() if q.startswith("trace") else None
                ax.errorbar(x, g.ratio_to_plain, yerr=yerr, fmt="o", ms=3, color=COLOR[m], elinewidth=.8, capsize=0)
                ax.plot([i - .28, i + .28], [g.ratio_to_plain.mean()] * 2, color=COLOR[m], lw=1.6)
            ax.set_xticks(range(3), ["R", "G", "RG"], fontsize=8)
            ax.set_xlim(-.6, 2.6)
            ax.grid(axis="y", alpha=.18)
            if r_ == 0:
                ax.set_title(lab, fontsize=8.5)
            if c_ == 0:
                ax.set_ylabel(("Training" if probe == "train_probe" else "Test") + " probe\nratio to Plain", fontsize=8.5)
            ax.tick_params(labelsize=7.5)
    cap = ("Within-seed ratio of each Hessian quantity to Plain: CIFAR-10 / ResNet-20-BN / SGD, final 30-epoch checkpoints, "
           "five training seeds (dots; bar = mean), 1k-image training (top) and test (bottom) probes. Mean cross-entropy without "
           "weight decay; centre-frozen BatchNorm (statistics recalibrated once at the unperturbed weights on 2,000 training "
           "images, then held fixed); variables are the 268,336 convolution and classifier weights. Eigenvalues: existing "
           "Lanczos estimates (converged, two starts). Traces: Hutchinson estimates with 128 Rademacher draws shared across "
           "methods within a seed; vertical bars are $\\pm1$ Monte Carlo SEM of the ratio (delta method, paired draws). "
           "$A_g=\\|\\theta_g\\|I$ per filter or classifier row; $C_g=\\|\\theta_g\\|^2/p_g\\,I$ is the covariance of the "
           "filter-normalised random directions, so $\\mathrm{tr}(HC)$ is the mean curvature those directions see. Traces are "
           "signed; negative eigenvalues exist.")
    save(fig, name, cap, ["tables/T_trace_paired_vs_plain.csv"])


def scales(name):
    pc = pd.read_csv(TAB / "T_trace_per_checkpoint.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, layout="constrained")
    for ax, probe in zip(axes, ("train_probe", "test_probe")):
        rows = [("r_units_extreme_top1", "steepest\ndirection"),
                ("r_units_isotropic_relative_mean", "isotropic,\nrelative coords"),
                ("trace_covariance", "our random\ndirections")]
        for j, (col, lab) in enumerate(rows):
            for i, m in enumerate(M):
                g = pc[(pc.probe == probe) & (pc.method == m)]
                y = g[col].to_numpy()
                x = j + (i - 1.5) * .16 + np.linspace(-.03, .03, len(y))
                ax.plot(x, y, "o", ms=2.8, color=COLOR[m], label=LONG[m] if j == 0 else None)
        ax.set_yscale("log")
        ax.set_xticks(range(3), [r[1] for r in rows], fontsize=7.5)
        ax.set_title("Training probe" if probe == "train_probe" else "Test probe", fontsize=9)
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Curvature per unit $r^2$ (nats)")
    axes[0].legend(fontsize=7, frameon=False, loc="center left")
    cap = ("Three averages of the same centre-frozen Hessian on one common scale, curvature per unit squared RMS relative "
           "block displacement $r$ (five seeds per method, log axis): the steepest direction ($G\\lambda_{\\max}$ of the "
           "block-relative Hessian, $G=698$ blocks), the average over directions isotropic in block-relative coordinates, and the "
           "average under our filter-normalised random-direction distribution. The steepest direction is about $10^4$ times "
           "steeper than the average random direction. CIFAR-10 / ResNet-20-BN / SGD final checkpoints, 1k-image probes.")
    save(fig, name, cap, ["tables/T_trace_per_checkpoint.csv"])


def main():
    ratios("F_curvature_ratios_to_plain")
    scales("F_curvature_scales_r_units")
    old = json.loads((FIG / "CAPTIONS.json").read_text()) if (FIG / "CAPTIONS.json").exists() else {}
    old.update(CAPTIONS)
    (FIG / "CAPTIONS.json").write_text(json.dumps(old, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

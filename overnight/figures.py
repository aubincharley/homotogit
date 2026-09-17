"""Figures and LaTeX tables from ``analysis/`` (run ``overnight.analyze`` first).

    py -m overnight.figures [--out <dir>]

PDF + PNG preview per figure, LaTeX tables and standalone captions.  Only the compact CSVs in
``analysis/`` are read, so the script runs unchanged inside the paper handoff.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ARMS = ("plain", "resolution_max_b1", "gaussian_postrelu", "resolution_max_b1_gaussian_conv",
        "cbs_published_schedule", "cbs_budget_matched", "sdpoint")
SHORT = {"plain": "Plain", "resolution_max_b1": "R", "gaussian_postrelu": "G", "resolution_max_b1_gaussian_conv": "RG",
         "cbs_published_schedule": "CBS-pub", "cbs_budget_matched": "CBS-bm", "sdpoint": "SDPoint"}
COLOR = {"plain": "#555555", "resolution_max_b1": "#1f77b4", "gaussian_postrelu": "#e08214", "resolution_max_b1_gaussian_conv": "#7b3294",
         "cbs_published_schedule": "#1a9850", "cbs_budget_matched": "#91cf60", "sdpoint": "#d7191c"}
CAMPAIGNS = ("historical_sgd_30", "historical_adamw_30", "sgd_standard_aug_160", "adamw_long_noaug_160", "adamw_long_aug_160")
CLABEL = {"historical_sgd_30": "SGD 0.005\n30 ep, no aug\n(historical)", "historical_adamw_30": "AdamW 0.02\n30 ep, no aug\n(historical)",
          "sgd_standard_aug_160": "SGD 0.1 steps\n160 ep, crop+flip", "adamw_long_noaug_160": "AdamW 0.02\n160 ep, no aug",
          "adamw_long_aug_160": "AdamW 0.02\n160 ep, crop+flip"}
CTEX = {"historical_sgd_30": "SGD 0.005, 30 ep, no aug.", "historical_adamw_30": "AdamW 0.02, 30 ep, no aug.",
        "sgd_standard_aug_160": "SGD 0.1 (steps), 160 ep, crop+flip", "adamw_long_noaug_160": "AdamW 0.02, 160 ep, no aug.",
        "adamw_long_aug_160": "AdamW 0.02, 160 ep, crop+flip"}
POLICIES = ("P0", "P1", "P2", "P3")
PMARK = {"P0": "x", "P1": "o", "P2": "s", "P3": "^"}
PLABEL = {"P0": "P0 saved buffers", "P1": "P1 cumulative, batch 500 (primary)", "P2": "P2 cumulative, batch 32",
          "P3": "P3 EMA, training loader"}


def save(fig, out, name):
    fig.savefig(out / (name + ".pdf"))
    fig.savefig(out / (name + ".png"), dpi=160)
    plt.close(fig)


def fig_r_sdpoint(pairs, out):
    sel = pairs[(pairs.split == "test") & pairs.contrast.isin(["R - SDPoint", "RG - SDPoint"])]
    camps = [c for c in CAMPAIGNS if c in set(sel.campaign)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, layout="constrained")
    for ax, name in zip(axes, ("R - SDPoint", "RG - SDPoint")):
        ax.axhline(0, color=".4", lw=.8)
        for i, camp in enumerate(camps):
            for j, pol in enumerate(POLICIES):
                r = sel[(sel.campaign == camp) & (sel.contrast == name) & (sel.policy == pol)]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = i + (j - 1.5) * 0.17
                seeds = [r["acc_diff_seed%d_pp" % s] for s in (0, 1, 2)]
                ax.scatter([x] * 3, seeds, s=9, color=".55", zorder=2, lw=0)
                ax.errorbar(x, r.acc_diff_mean_pp, yerr=r.acc_diff_sd_pp, fmt=PMARK[pol], ms=6, color="C%d" % j,
                            mfc="white" if pol != "P1" else "C1", capsize=2, lw=1, zorder=3)
        ax.set_xticks(range(len(camps)))
        ax.set_xticklabels([CLABEL[c] for c in camps], fontsize=7)
        ax.set_title("%s, test accuracy (pp, seed-paired)" % name, fontsize=9)
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("difference (pp)")
    h = [plt.Line2D([], [], marker=PMARK[p], ls="", color="C%d" % j, mfc="white" if p != "P1" else "C1", label=PLABEL[p])
         for j, p in enumerate(POLICIES)] + [plt.Line2D([], [], marker="o", ls="", ms=3, color=".55", label="individual seeds")]
    fig.legend(handles=h, loc="outside lower center", ncol=5, fontsize=7, frameon=False)
    save(fig, out, "fig_r_rg_vs_sdpoint_by_policy")


def fig_cost(summ, costs, out):
    regimes = [c for c in CAMPAIGNS[2:] if c in set(costs.campaign)]
    if not regimes:
        return
    fig, axes = plt.subplots(1, len(regimes), figsize=(3.6 * len(regimes), 3.3), layout="constrained", squeeze=False)
    for ax, camp in zip(axes[0], regimes):
        for arm in ARMS:
            c = costs[(costs.campaign == camp) & (costs.arm == arm)]
            s = summ[(summ.campaign == camp) & (summ.arm == arm) & (summ.policy == "P1") & (summ.split == "test")]
            if c.empty or s.empty:
                continue
            ax.errorbar(c.training_gpu_hours.mean(), s.acc_mean_pct.iloc[0], yerr=s.acc_sd_pct.iloc[0],
                        xerr=c.training_gpu_hours.std(ddof=1) if len(c) > 1 else None, fmt="o", color=COLOR[arm], capsize=2)
            ax.annotate(SHORT[arm], (c.training_gpu_hours.mean(), s.acc_mean_pct.iloc[0]), fontsize=7,
                        xytext=(4, 3), textcoords="offset points", color=COLOR[arm])
        ax.set_title(CTEX[camp], fontsize=9)
        ax.set_xlabel("training loop, T4 GPU-hours per run")
        ax.grid(alpha=.25)
    axes[0][0].set_ylabel("P1 test accuracy (%), mean ± SD, 3 seeds")
    save(fig, out, "fig_accuracy_vs_training_cost")


def fig_curves(p1, saved, out):
    regimes = [c for c in CAMPAIGNS[2:] if c in set(p1.campaign)]
    if not regimes:
        return
    fig, axes = plt.subplots(2, len(regimes), figsize=(3.8 * len(regimes), 6.2), layout="constrained", squeeze=False, sharex=True)
    for k, camp in enumerate(regimes):
        ax, ax2 = axes[0][k], axes[1][k]
        for arm in ARMS:
            g = p1[(p1.campaign == camp) & (p1.arm == arm)].groupby("epoch").test_acc.agg(["mean", "std"])
            if not g.empty:
                ax.errorbar(g.index, 100 * g["mean"], yerr=100 * g["std"], marker="o", ms=4, color=COLOR[arm], label=SHORT[arm], capsize=2, lw=1.2)
            s = saved[(saved.campaign == camp) & (saved.arm == arm)].groupby("epoch").current_test_acc.mean()
            if not s.empty:
                ax2.plot(s.index, 100 * s.values, color=COLOR[arm], lw=.9, ls="--" if arm == "sdpoint" else "-", label=SHORT[arm])
        for a in (ax, ax2):
            for e in (32, 64):
                a.axvline(e, color="#1f77b4", lw=.6, ls=":")
            a.axvline(112, color="#e08214", lw=.6, ls=":")
            if camp.startswith("sgd"):
                for e in (80, 120):
                    a.axvline(e, color="k", lw=.6, ls="-.")
            a.grid(alpha=.2)
        ax.set_title(CTEX[camp], fontsize=9)
        ax2.set_xlabel("completed epochs")
        # the first epochs would otherwise compress the whole diagnostic panel
        late = saved[(saved.campaign == camp) & (saved.epoch >= 10)].current_test_acc
        if len(late):
            ax2.set_ylim(100 * late.min() - 1, 100 * late.max() + 1)
    axes[0][0].set_ylabel("test acc (%), P1 at scheduled state")
    axes[1][0].set_ylabel("test acc (%), saved buffers, current path")
    axes[0][-1].legend(fontsize=7, frameon=False, loc="lower right", ncol=2)
    # guides are described in CAPTIONS.md rather than on the figure, which has no free space:
    # dotted blue = R transitions (32, 64), dotted orange = G/RG bypass (112), dash-dot = SGD lr steps
    save(fig, out, "fig_learning_curves_p1_and_saved")


def fig_gain_heatmap(pairs, out):
    sel = pairs[(pairs.split == "test") & (pairs.policy == "P1") & pairs.contrast.str.endswith("- Plain")]
    camps = [c for c in CAMPAIGNS if c in set(sel.campaign)]
    arms = [a for a in ARMS if a != "plain"]
    M = np.full((len(arms), len(camps)), np.nan)
    T = [["" for _ in camps] for _ in arms]
    for i, a in enumerate(arms):
        for j, c in enumerate(camps):
            r = sel[(sel.campaign == c) & (sel.arm_a == a)]
            if not r.empty:
                r = r.iloc[0]
                M[i, j] = r.acc_diff_mean_pp
                T[i][j] = "%+.2f\n(%d+/%d-)" % (r.acc_diff_mean_pp, r.acc_n_positive, r.acc_n_negative)
    fig, ax = plt.subplots(figsize=(1.6 * len(camps) + 1.5, 4), layout="constrained")
    lim = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(arms)):
        for j in range(len(camps)):
            ax.text(j, i, T[i][j], ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(camps)))
    ax.set_xticklabels([CLABEL[c] for c in camps], fontsize=7)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([SHORT[a] for a in arms])
    fig.colorbar(im, ax=ax, label="P1 test accuracy − Plain (pp, mean of 3 seed pairs)")
    save(fig, out, "fig_gain_vs_plain_p1")


def tex_tables(summ, pairs, order, out):
    lines = [r"\begin{tabular}{l" + "c" * 5 + "}", r"\toprule",
             "Arm & " + " & ".join(CTEX[c] for c in CAMPAIGNS) + r" \\", r"\midrule"]
    for arm in ARMS:
        cells = []
        for c in CAMPAIGNS:
            r = summ[(summ.campaign == c) & (summ.arm == arm) & (summ.policy == "P1") & (summ.split == "test")]
            cells.append("--" if r.empty else "%.2f $\\pm$ %.2f" % (r.acc_mean_pct.iloc[0], r.acc_sd_pct.iloc[0]))
        lines.append("%s & %s \\\\" % (SHORT[arm], " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_absolute_test_accuracy_P1.tex").write_text("\n".join(lines) + "\n")
    names = ["R - Plain", "G - Plain", "RG - Plain", "CBS-pub - Plain", "CBS-bm - Plain", "SDPoint - Plain", "R - SDPoint", "RG - SDPoint",
             "G - CBS-pub", "G - CBS-bm", "RG - R", "CBS-bm - CBS-pub"]
    lines = [r"\begin{tabular}{l" + "c" * 5 + "}", r"\toprule", "Contrast & " + " & ".join(CTEX[c] for c in CAMPAIGNS) + r" \\", r"\midrule"]
    for n in names:
        cells = []
        for c in CAMPAIGNS:
            r = pairs[(pairs.campaign == c) & (pairs.contrast == n) & (pairs.policy == "P1") & (pairs.split == "test")]
            cells.append("--" if r.empty else "%+.2f $\\pm$ %.2f (%d/%d)" % (r.acc_diff_mean_pp.iloc[0], r.acc_diff_sd_pp.iloc[0],
                                                                           r.acc_n_positive.iloc[0], r.acc_n_negative.iloc[0]))
        lines.append("%s & %s \\\\" % (n.replace(" - ", " $-$ "), " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_paired_contrasts_P1.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{llcccc}", r"\toprule", r"Recipe & Contrast & P0 & P1 & P2 & P3 \\", r"\midrule"]
    for c in CAMPAIGNS:
        for n in ("R - SDPoint", "RG - SDPoint"):
            r = order[(order.campaign == c) & (order.contrast == n)]
            if r.empty:
                continue
            r = r.iloc[0]
            cells = ["%+.2f (%s)" % (r["%s_mean_pp" % p], r["%s_signs(+/-)" % p]) if ("%s_mean_pp" % p) in r and pd.notna(r["%s_mean_pp" % p]) else "--"
                     for p in POLICIES]
            lines.append("%s & %s & %s \\\\" % (CTEX[c], n.replace(" - ", " $-$ "), " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_r_rg_vs_sdpoint_by_policy.tex").write_text("\n".join(lines) + "\n")


CAPTIONS = """# Captions (standalone)

**fig_r_rg_vs_sdpoint_by_policy.** Seed-paired test-accuracy differences R − SDPoint (left) and RG − SDPoint (right) on CIFAR-10 / ResNet-20, for five training recipes and four BatchNorm evaluation policies at each method's native inference state. P0: checkpoint buffers (for SDPoint these mix training instances). P1 (primary): buffers reset and re-estimated as a cumulative average over the 50,000 clean training images, batch 500. P2: the same with batch 32. P3: author-style EMA pass (momentum 0.1, no reset) over a training loader with the recipe's batch size and augmentation. Markers show the mean ± SD of 3 seed-paired differences; grey dots are individual seeds. The historical 30-epoch recipes had prior test-set exposure.

**fig_accuracy_vs_training_cost.** P1 test accuracy (mean ± SD, 3 seeds) against measured training-loop time per run (T4 GPU-hours; excludes periodic evaluation, diagnostics and final calibration) for the three 160-epoch recipes. Published-schedule CBS also keeps 19 extra 3×3 depthwise convolutions at inference. After 160 epochs its σ = 0.9³¹, and these kernels are numerically identities.

**fig_learning_curves_p1_and_saved.** Top: P1 test accuracy after completed epochs 40, 80, 120 and 160, evaluated at each method's scheduled state at that epoch (SDPoint: full-resolution instance). Early points therefore describe a different network from the final one for R/G/RG/CBS. Bottom: the per-epoch saved-buffer evaluation on the current path (diagnostic). SDPoint's saved-buffer curve is not its calibrated performance.

**fig_gain_vs_plain_p1.** Mean seed-paired P1 test-accuracy gain over Plain (pp) for every arm and recipe, with sign counts over the 3 seeds.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    here = Path(__file__).resolve().parents[1] / "studies" / "overnight_long_aug_bn"
    src = Path(a.analysis) if a.analysis else here / "analysis"
    out = Path(a.out) if a.out else here / "figures"
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(src / "paired_contrasts.csv")
    summ = pd.read_csv(src / "summary_by_arm.csv")
    fig_r_sdpoint(pairs, out)
    fig_gain_heatmap(pairs, out)
    costs = pd.read_csv(src / "costs.csv") if (src / "costs.csv").stat().st_size > 2 else pd.DataFrame(columns=["campaign"])
    fig_cost(summ, costs, out)
    p1 = pd.read_csv(src / "learning_curves_p1.csv") if (src / "learning_curves_p1.csv").stat().st_size > 2 else pd.DataFrame(columns=["campaign"])
    saved = pd.read_csv(src / "learning_curves_saved_bn_per_epoch.csv") if (src / "learning_curves_saved_bn_per_epoch.csv").stat().st_size > 2 else pd.DataFrame(columns=["campaign"])
    fig_curves(p1, saved, out)
    tex_tables(summ, pairs, pd.read_csv(src / "r_rg_vs_sdpoint_across_policies.csv"), out)
    (out / "CAPTIONS.md").write_text(CAPTIONS, encoding="utf-8")
    print("figures ->", out)


if __name__ == "__main__":
    main()

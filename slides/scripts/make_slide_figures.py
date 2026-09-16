"""Figures for the presentation slides.

    py -3 scripts/make_slide_figures.py [PATH_TO_paper]

Reads only the audited tables of the paper project (``paper/data/records``);
no training, evaluation or interpolation is performed. Method colours match
the paper figures. Outputs PNG files (300 dpi) into ``figures/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parents[1]
PAPER = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "paper"
REC = PAPER / "data" / "records"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- style
# Computer Modern Sans (shipped with matplotlib) matches the Beamer default font.
INK, MUTED, GRID = "#1d2433", "#6b7385", "#e6e8ee"
plt.rcParams.update({
    "font.family": "cmss10", "font.size": 11, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "savefig.dpi": 300, "mathtext.fontset": "cm",
    "axes.unicode_minus": False, "axes.formatter.use_mathtext": False,
})

M = ["plain", "resolution_max_b1", "gaussian_postrelu", "resolution_max_b1_gaussian_conv"]
SHORT = dict(zip(M, ["Plain", "R", "G", "RG"]))
LONG = dict(zip(M, ["Plain", "Resolution", "Gaussian", "Combined"]))
COLOR = dict(zip(M, ["#333333", "#1976b2", "#dc7b13", "#8653aa"]))


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), bbox_inches="tight", pad_inches=0.04, transparent=False,
                facecolor="white")
    plt.close(fig)
    print("wrote figures/%s.png" % name)


# ---------------------------------------------------------------- schedules
def fig_schedules():
    e = np.arange(30)
    r = np.where(e < 6, 16, np.where(e < 12, 24, 32))
    plateau = [1.00, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30]
    g = np.array([plateau[min(k // 3, 6)] if k < 21 else 0.0 for k in e])
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 2.9))
    ax[0].step(np.append(e, 30), np.append(r, 32), where="post", color=COLOR[M[1]], lw=2.4)
    ax[0].set_yticks([16, 24, 32]); ax[0].set_ylim(12, 35)
    ax[0].set_title(r"Feature-map side length $r(e)$", loc="left")
    ax[1].step(np.append(e, 30), np.append(g, 0), where="post", color=COLOR[M[2]], lw=2.4,
               label=r"G: $\sigma = g(e)$")
    ax[1].step(np.append(e, 30), np.append(g * r / 32, 0), where="post", color=COLOR[M[3]],
               lw=2.4, ls=(0, (4, 1.6)), label=r"RG: $\sigma = g(e)\,r(e)/32$")
    ax[1].set_ylim(-0.05, 1.12); ax[1].set_yticks([0, .5, 1])
    ax[1].set_title(r"Gaussian scale $\sigma(e)$", loc="left")
    ax[1].legend(frameon=False, fontsize=10, loc="upper right")
    for a in ax:
        a.set_xlim(0, 30); a.set_xticks([0, 6, 12, 21, 30]); a.set_xlabel("epoch")
        a.grid(axis="y", color=GRID, lw=.8); a.set_axisbelow(True)
    fig.tight_layout(w_pad=3)
    save(fig, "schedules")


# ---------------------------------------------------------------- selection batch
def fig_selection(S):
    s = S[S.setting == "sgd"].set_index("method").loc[M]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(4)[::-1]
    for i, m in enumerate(M):
        ax[0].barh(y[i], s.test_acc_mean[m], xerr=s.test_acc_sd[m], color=COLOR[m], height=.62,
                   error_kw=dict(ecolor=MUTED, lw=1.1, capsize=3))
        ax[0].text(s.test_acc_mean[m] + s.test_acc_sd[m] + .25, y[i], "%.2f" % s.test_acc_mean[m],
                   va="center", fontsize=11, color=INK)
        ax[1].barh(y[i], s.wall_minutes_mean[m], xerr=s.wall_minutes_sd[m], color=COLOR[m],
                   height=.62, alpha=.85, error_kw=dict(ecolor=MUTED, lw=1.1, capsize=3))
        ax[1].text(s.wall_minutes_mean[m] + s.wall_minutes_sd[m] + .2, y[i],
                   "%.1f min" % s.wall_minutes_mean[m], va="center", fontsize=11, color=INK)
    ax[0].set_xlim(70, 84); ax[0].set_xlabel("test accuracy (%)")
    ax[0].set_yticks(y); ax[0].set_yticklabels([LONG[m] for m in M], fontsize=11.5)
    ax[1].set_xlim(0, 15.5); ax[1].set_xlabel("training time (min, T4)")
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    for a in ax:
        a.grid(axis="x", color=GRID, lw=.8); a.set_axisbelow(True)
        a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=2)
    save(fig, "selection")


# ---------------------------------------------------------------- transfer gains
SETTINGS = [
    ("sgd", "CIFAR-10, SGD"), ("adam", "CIFAR-10, Adam"), ("adamw", "CIFAR-10, AdamW"),
    ("radam", "CIFAR-10, RAdam"), ("resnet20act_cifar10/gelu", "CIFAR-10, GELU"),
    ("resnet20act_cifar10/silu", "CIFAR-10, SiLU"), ("resnet20gn_cifar10", "CIFAR-10, GroupNorm"),
    ("vgg11_cifar10", "CIFAR-10, VGG-11"), ("svhn", "SVHN"), ("stl10", "STL-10 (96 px)"),
    ("cifar10_5k/stl_budget", "CIFAR-10 5k, 60 ep"),
    ("cifar10_5k/reference_updates", "CIFAR-10 5k, 293 ep"),
]


def fig_gains(S):
    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    n = len(SETTINGS)
    ypos = np.arange(n)[::-1]
    ax.axvline(0, color=MUTED, lw=1)
    for k, (key, label) in enumerate(SETTINGS):
        if k % 2 == 0:
            ax.axhspan(ypos[k] - .5, ypos[k] + .5, color="#f5f6f9", lw=0, zorder=0)
        for j, m in enumerate(M[1:]):
            row = S[(S.setting == key) & (S.method == m)].iloc[0]
            yy = ypos[k] + (1 - j) * .24
            ax.errorbar(row.gain_mean, yy, xerr=row.gain_sd, fmt="o", ms=6.5, color=COLOR[m],
                        ecolor=COLOR[m], elinewidth=1.3, capsize=0, zorder=3)
    ax.set_yticks(ypos); ax.set_yticklabels([l for _, l in SETTINGS], fontsize=12)
    ax.set_ylim(-.6, n - .4)
    ax.set_xlabel("gain over Plain, same seed (accuracy points)\n" r"mean $\pm$ SD over 3 seeds")
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], color=COLOR[m], marker="o", ls="", ms=7, label=LONG[m]) for m in M[1:]]
    ax.legend(handles=h, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.4, 1.0),
              fontsize=12)
    fig.tight_layout()
    save(fig, "transfer_gains")


# ---------------------------------------------------------------- train vs test CE
def fig_train_test(E):
    e = E[E.policy == "saved"]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for m in M:
        g = e[e.method == m]
        ax.scatter(g.train_full_ce, g.test_full_ce, s=46, color=COLOR[m], label=LONG[m],
                   edgecolor="white", lw=.7, zorder=3)
        ax.scatter(g.train_full_ce.mean(), g.test_full_ce.mean(), s=190, marker="D",
                   color=COLOR[m], edgecolor="white", lw=1.4, zorder=4)
    ax.set_xlabel("training CE, all 50 000 images (nats)")
    ax.set_ylabel("test CE, 10 000 images (nats)")
    ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    fig.tight_layout()
    save(fig, "train_test_ce")


# ---------------------------------------------------------------- surfaces
def surfaces(planes):
    g0 = planes[(planes.seed == 0) & (planes.grid == 41)]
    Z = {}
    for m in M:
        g = g0[g0.method == m]
        c = float(g[(g.a == 0) & (g.b == 0)].train_probe_ce.iloc[0])
        z = g.pivot(index="b", columns="a", values="train_probe_ce").sort_index().sort_index(axis=1) - c
        assert z.shape == (41, 41) and not z.isna().any().any()
        Z[m] = z
    lo = min(z.to_numpy().min() for z in Z.values()); hi = max(z.to_numpy().max() for z in Z.values())
    norm = Normalize(lo, hi)

    def draw(ax, m, title=True):
        z = Z[m]
        xx, yy = np.meshgrid(z.columns, z.index)
        # measured vertices only: no fit, no smoothing, no new evaluation
        ax.plot_surface(xx, yy, z.to_numpy(), cmap="viridis", norm=norm, rstride=1, cstride=1,
                        linewidth=.16, edgecolor=(0, 0, 0, .16), antialiased=True, shade=False)
        ax.set_xlim(-.5, .5); ax.set_ylim(-.5, .5); ax.set_zlim(lo, hi)
        ax.set_xticks([-.5, 0, .5]); ax.set_yticks([0, .5]); ax.set_zticks([0, 2, 4])
        ax.tick_params(labelsize=9, pad=-2)
        ax.set_xlabel("a", fontsize=10, labelpad=-4); ax.set_ylabel("b", fontsize=10, labelpad=-4)
        ax.set_box_aspect((1, 1, .9)); ax.view_init(elev=25, azim=-55)
        if title:
            ax.set_title(LONG[m], fontsize=15, color=COLOR[m], pad=-18, fontweight="medium")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis._axinfo["grid"]["linewidth"] = .35
            axis._axinfo["grid"]["color"] = (.75, .75, .75, .5)

    def colorbar(fig, rect):
        cax = fig.add_axes(rect)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax,
                          orientation="horizontal", ticks=[0, 1, 2, 3, 4])
        cb.set_label("CE above the trained point (nats)", fontsize=10)
        cb.ax.tick_params(labelsize=9, length=2); cb.outline.set_visible(False)

    # one large surface per method
    for m in M:
        fig = plt.figure(figsize=(4.4, 4.1))
        ax = fig.add_axes([-.1, .16, 1.2, .98], projection="3d")
        draw(ax, m)
        colorbar(fig, [.22, .06, .56, .03])
        save(fig, "surface_" + SHORT[m].lower())

    # measured cross-sections through the centre (b = 0 and a = 0), per method
    for m in M[1:]:
        fig, ax = plt.subplots(1, 2, figsize=(4.4, 2.1), sharey=True)
        for k, (axis_name, sel) in enumerate((("a", "b"), ("b", "a"))):
            for mm, lw, z in ((M[0], 1.8, 1), (m, 2.4, 2)):
                zz = Z[mm]
                prof = zz.loc[0.0] if axis_name == "a" else zz[0.0]
                ax[k].plot(prof.index, prof.values, color=COLOR[mm], lw=lw, zorder=z,
                           label=LONG[mm])
            ax[k].set_xlabel("%s  (%s = 0)" % (axis_name, sel), fontsize=9.5)
            ax[k].set_xticks([-.5, 0, .5]); ax[k].tick_params(labelsize=8.5)
            ax[k].grid(color=GRID, lw=.7); ax[k].set_axisbelow(True)
        ax[0].set_ylabel("CE rise (nats)", fontsize=9.5)
        ax[0].legend(frameon=False, fontsize=8.5, loc="upper center")
        fig.tight_layout(w_pad=.8)
        save(fig, "slice_" + SHORT[m].lower())

    rise = {m: float(Z[m].to_numpy().mean()) for m in M}
    return rise


# ---------------------------------------------------------------- Hessian
def fig_hessian(H):
    h = H[(H.policy == "centre_frozen") & (H.coords == "relative")]
    plain = h[h.method == "plain"].set_index(["seed", "probe"]).top1
    fig, ax = plt.subplots(figsize=(4.9, 3.1))
    width = .26
    out = {}
    for j, m in enumerate(M[1:]):
        g = h[h.method == m].set_index(["seed", "probe"]).top1
        ratio = (g / plain).reset_index()
        for i, probe in enumerate(("train_probe", "test_probe")):
            r = ratio[ratio.probe == probe].top1
            out[(m, probe)] = (r.mean(), r.std(ddof=1), int((r < 1).sum()))
            ax.bar(i + (j - 1) * width, r.mean(), width * .9, yerr=r.std(ddof=1), color=COLOR[m],
                   error_kw=dict(ecolor=MUTED, lw=1, capsize=2.5), label=LONG[m] if i == 0 else None)
    ax.axhline(1, color=COLOR["plain"], lw=1.4, ls=(0, (4, 2)))
    ax.text(-.46, 1.03, "Plain = 1", va="bottom", fontsize=10, color=COLOR["plain"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["training probe", "test probe"], fontsize=11)
    ax.set_ylim(0, 1.18); ax.set_xlim(-.5, 1.5)
    ax.set_ylabel(r"$\lambda_{\max}\,/\,\lambda_{\max}^{\mathrm{Plain}}$")
    ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, ncol=3, loc="lower center", bbox_to_anchor=(.45, 1.0))
    fig.tight_layout()
    save(fig, "hessian")
    return out


# ---------------------------------------------------------------- benchmark table
TABLE = [
    ("CIFAR-10", "ResNet-20 / BN / ReLU", "SGD", "sgd"),
    ("CIFAR-10", "ResNet-20 / BN / ReLU", "Adam", "adam"),
    ("CIFAR-10", "ResNet-20 / BN / ReLU", "AdamW", "adamw"),
    ("CIFAR-10", "ResNet-20 / BN / ReLU", "RAdam", "radam"),
    ("CIFAR-10", "ResNet-20 / BN / GELU", "SGD", "resnet20act_cifar10/gelu"),
    ("CIFAR-10", "ResNet-20 / BN / SiLU", "SGD", "resnet20act_cifar10/silu"),
    ("CIFAR-10", "ResNet-20 / GN / ReLU", "SGD", "resnet20gn_cifar10"),
    ("CIFAR-10", "VGG-11 / BN / ReLU", "SGD", "vgg11_cifar10"),
    ("SVHN", "ResNet-20 / BN / ReLU", "SGD", "svhn"),
    ("STL-10", "ResNet-20 / BN / ReLU", "SGD", "stl10"),
    ("CIFAR-10 (5k)", "ResNet-20 / BN / ReLU", "SGD", "cifar10_5k/stl_budget"),
    ("CIFAR-10 (5k)", "ResNet-20 / BN / ReLU", "SGD", "cifar10_5k/reference_updates"),
]


def benchmark_table(S):
    lines = ["% GENERATED by scripts/make_slide_figures.py from paper/data/records/benchmark_settings.csv"]
    for k, (data, model, opt, key) in enumerate(TABLE):
        rows = S[S.setting == key].set_index("method")
        epochs = int(rows.epochs.iloc[0])
        means = {m: rows.test_acc_mean[m] for m in M}
        top = "%.1f" % max(means.values())
        cells = []
        for m in M:
            v = "%.1f" % means[m]
            sd = "%.1f" % rows.test_acc_sd[m]
            sd = r"$<$0.1" if sd == "0.0" else r"$\pm$" + sd
            if v == top:  # ties at the displayed precision are all marked
                v = r"\best{%s}" % v
            elif m != "plain" and means[m] < means["plain"]:
                v = r"\worse{%s}" % v
            cells.append(r"%s\sd{%s}" % (v, sd))
        if k in (8, 9, 10):  # dataset changes: SVHN, STL-10, CIFAR-10 5k
            lines.append(r"\midrule")
        lines.append(r"%s & %s & %s & %d & %s \\" % (data, model, opt, epochs, " & ".join(cells)))
    (OUT / "benchmark_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote figures/benchmark_rows.tex")


def main():
    S = pd.read_csv(REC / "benchmark_settings.csv")
    fig_schedules()
    fig_selection(S)
    fig_gains(S)
    benchmark_table(S)
    fig_train_test(pd.read_csv(REC / "landscape_endpoints.csv"))
    rise = surfaces(pd.read_csv(REC / "landscape_random_planes.csv"))
    hess = fig_hessian(pd.read_csv(REC / "landscape_hessian.csv"))
    print("mean CE rise over the 41x41 plane, seed 0:", {SHORT[m]: round(v, 3) for m, v in rise.items()})
    print("Hessian ratio (centre-frozen, relative):",
          {(SHORT[m], p): (round(a, 2), round(b, 2), c) for (m, p), (a, b, c) in hess.items()})


if __name__ == "__main__":
    main()

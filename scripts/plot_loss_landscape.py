"""Draw the filtered-vs-unfiltered landscape slice written by
``scripts/loss_landscape_blur.py``.

Everything is drawn on a log scale.  With filter-normalized directions the
cross-entropy spans four orders of magnitude across a span-1 slice, so a linear
colour scale shows one flat blob and nothing else; log10 is the standard choice
for these plots and makes both the basin and the walls readable at once.

A  the two surfaces on a **shared** log scale -- comparable directly, because
   the log compresses the level offset instead of letting it hide the shape.
B  their ratio, log10(L_0 / L_sigma): scale-free, so it reports where the two
   objectives disagree rather than where the loss happens to be large.
C  1-D cuts through the anchor along both directions, log10 loss against
   distance, showing level and width together.

Everything is read from the npz; nothing is recomputed here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "landscape"
BLUE, RED, GREY = "#1f77b4", "#d62728", "#7a7a7a"

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 10, "legend.fontsize": 9,
                     "figure.facecolor": "white"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default=None)
    a = ap.parse_args()
    # newest by mtime: lexicographic order puts "_g5_" after "_g21_"
    path = (Path(a.npz) if a.npz else
            max(OUT.glob("landscape_*.npz"), key=lambda q: q.stat().st_mtime))
    d = np.load(path, allow_pickle=True)
    xs, ys = d["xs"], d["ys"]
    blur, plain = d["loss_blur"], d["loss_plain"]
    sigma, epoch = float(d["sigma"]), int(d["epoch"])
    n_img, arch = int(d["n_images"]), str(d["arch"])
    mid = len(xs) // 2

    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.42,
                          wspace=0.32, left=0.07, right=0.97, top=0.88,
                          bottom=0.09)

    lb, lp = np.log10(blur), np.log10(plain)
    lo, hi = min(lb.min(), lp.min()), max(lb.max(), lp.max())
    lv = np.linspace(lo, hi, 20)
    for k, (surf, name) in enumerate((
            (lb, r"filtered:  $L_\sigma(\theta)$, $\sigma=%.2f$" % sigma),
            (lp, r"unfiltered:  $L_0(\theta)$, filters bypassed"))):
        ax = fig.add_subplot(gs[0, k])
        cf = ax.contourf(xs, ys, surf, levels=lv, cmap="viridis")
        ax.contour(xs, ys, surf, levels=lv[::3], colors="w", linewidths=0.4,
                   alpha=0.55)
        ax.plot(0, 0, "o", ms=5, mfc="w", mec="k", mew=1.2)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(r"$\alpha$")
        if k == 0:
            ax.set_ylabel(r"$\beta$")
        cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"$\log_{10} L$", fontsize=8)

    # where the two objectives disagree, as a scale-free ratio
    ax = fig.add_subplot(gs[0, 2])
    ratio = lp - lb
    m = float(np.abs(ratio).max())
    cf = ax.contourf(xs, ys, ratio, levels=np.linspace(-m, m, 20), cmap="RdBu_r")
    ax.contour(xs, ys, ratio, levels=8, colors="k", linewidths=0.3, alpha=0.35)
    ax.plot(0, 0, "o", ms=5, mfc="w", mec="k", mew=1.2)
    ax.set_title(r"$\log_{10}(L_0 / L_\sigma)$", fontsize=10)
    ax.set_xlabel(r"$\alpha$")
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)

    # 1-D cuts, each shifted to its own centre: comparing width, not level
    axc = fig.add_subplot(gs[1, :2])
    for surf, lab, col in ((blur, r"$L_\sigma$ (filtered)", BLUE),
                           (plain, r"$L_0$ (unfiltered)", RED)):
        axc.plot(xs, surf[mid, :], color=col, lw=2.0,
                 label=r"%s,  centre $=%.3f$" % (lab, surf[mid, mid]))
        axc.plot(ys, surf[:, mid], color=col, lw=1.0, ls="--", alpha=0.75)
    axc.set_yscale("log")
    axc.axvline(0, color="0.8", lw=0.8)
    axc.set_xlabel(r"distance along $\alpha$ (solid) and $\beta$ (dashed)")
    axc.set_ylabel(r"cross-entropy (log)")
    axc.set_title("1-D cuts through the anchor, both directions", fontsize=10)
    axc.legend(frameon=False, loc="upper center")

    # curvature at the anchor: the quantity the smoothing claim is about
    h = xs[1] - xs[0]
    curv = {}
    for nm, surf in (("blur", blur), ("plain", plain)):
        curv[nm] = ((surf[mid, mid + 1] - 2 * surf[mid, mid] + surf[mid, mid - 1]) / h ** 2,
                    (surf[mid + 1, mid] - 2 * surf[mid, mid] + surf[mid - 1, mid]) / h ** 2)
    jb = np.unravel_index(blur.argmin(), blur.shape)
    jp = np.unravel_index(plain.argmin(), plain.shape)

    axt = fig.add_subplot(gs[1, 2])
    axt.axis("off")
    axt.text(0, 1.0, "\n".join([
        r"anchor: epoch %d checkpoint" % epoch,
        r"arch: %s" % arch.replace("_", r"\_"),
        r"$\sigma = %.2f$ at every one of the 19 sites" % sigma,
        r"($q_l = 1$: this arm is R32)",
        "",
        r"%d test images, fixed across the grid" % n_img,
        r"grid %d$\times$%d, span $\pm$%.1f" % (len(xs), len(ys), xs[-1]),
        r"directions: filter-normalised, seed %d" % int(d["seed"]),
        "",
        r"centre $L_\sigma = %.3f$" % float(d["centre_blur"]),
        r"centre $L_0 = %.3f$  ($\times%.2f$)"
        % (float(d["centre_plain"]),
           float(d["centre_plain"]) / float(d["centre_blur"])),
        "",
        r"curvature at the anchor",
        r"  $L_\sigma$: %.0f ($\alpha$), %.0f ($\beta$)" % curv["blur"],
        r"  $L_0$:      %.0f ($\alpha$), %.0f ($\beta$)" % curv["plain"],
        r"  filtered is %.1f$\times$ / %.1f$\times$ flatter"
        % (curv["plain"][0] / curv["blur"][0], curv["plain"][1] / curv["blur"][1]),
        "",
        r"slice minimum",
        r"  $L_\sigma$ at $(%.2f, %.2f)$" % (xs[jb[1]], ys[jb[0]]),
        r"  $L_0$ at $(%.2f, %.2f)$" % (xs[jp[1]], ys[jp[0]]),
    ]), va="top", ha="left", fontsize=8.5, family="monospace",
        transform=axt.transAxes)

    fig.suptitle("Loss-landscape slice at one set of weights: the same "
                 "directions, two objectives", fontsize=12)
    stem = path.stem.replace("landscape_", "")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / ("landscape_%s.%s" % (stem, ext)), dpi=160,
                    bbox_inches="tight")
    plt.close(fig)
    print("read ", path)
    print("wrote", OUT / ("landscape_%s.png" % stem))


if __name__ == "__main__":
    main()

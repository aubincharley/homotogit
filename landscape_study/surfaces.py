"""3-D renderings of the already evaluated grids (no new evaluation).

    py -m landscape_study.surfaces --results studies/landscape_v1/results

Every surface vertex is a measured grid point from ``results/``; faces are the
standard bilinear patches between neighbouring vertices.  No smoothing, no
extrapolation, no added points.  Within one comparison all panels share the
coordinate range, camera, box aspect, vertical limits and colour scale (train
and test have separate scales).  Centre-subtracted surfaces keep negative values
with a diverging scale centred at 0.  The floor of each box carries the contour
lines of the same grid.

Outputs (``studies/landscape_v1/figures/``, originals untouched):
``A_surface3d_absolute``, ``A_surface3d_centered``, ``B_surface3d_absolute``,
``B_surface3d_centered``, ``C_surface3d_plane``, ``C_checkpoint_ce`` (PDF+PNG)
and ``surfaces_interactive.html``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, colors

from .figures import FIG, INPUTS, center_of, grid_array, load, save

ELEV, AZIM, BOX = 28, -58, (1, 1, 0.75)
SPLIT = {"train_probe": "train probe (1,000 imgs)", "test_probe": "test probe (1,000 imgs)"}
BN = "recalibrated BatchNorm"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5})


def surface(ax, a, b, Z, norm, cmap, zlim, title, zlabel):
    A, B = np.meshgrid(a, b)
    ax.plot_surface(A, B, Z, facecolors=cmap(norm(Z)), rstride=1, cstride=1,
                    linewidth=0.15, edgecolor=(0, 0, 0, 0.25), antialiased=True, shade=False)
    ax.contour(A, B, Z, levels=8, zdir="z", offset=zlim[0], cmap=cmap, norm=norm, linewidths=0.7)
    ax.set_xlim(a[0], a[-1]); ax.set_ylim(b[0], b[-1]); ax.set_zlim(*zlim)
    ax.set_box_aspect(BOX)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel("a (dir. D)", labelpad=1); ax.set_ylabel("b (dir. E)", labelpad=1)
    if zlabel:
        ax.set_zlabel(zlabel, labelpad=2)
    ax.tick_params(pad=0, labelsize=6.5)
    ax.set_title(title, pad=0)


def grid_figure(panels, rows, cols, name, suptitle, centered, figsize):
    """panels: dict (row, col) -> (a, b, Z, title); scales shared per row (= split)."""
    fig = plt.figure(figsize=figsize)
    for r in range(rows):
        Zs = [panels[(r, c)][2] for c in range(cols)]
        lo, hi = min(z.min() for z in Zs), max(z.max() for z in Zs)
        if centered:
            m = max(abs(lo), abs(hi))
            norm, cmap = colors.TwoSlopeNorm(0.0, -m, m), cm.RdBu_r
            zlim = (lo - 0.05 * (hi - lo), hi)
            zlabel = "CE - centre CE"
        else:
            norm, cmap = colors.Normalize(lo, hi), cm.viridis
            zlim = (lo - 0.05 * (hi - lo), hi)
            zlabel = "cross-entropy"
        for c in range(cols):
            a, b, Z, title = panels[(r, c)]
            ax = fig.add_subplot(rows, cols, r * cols + c + 1, projection="3d")
            surface(ax, a, b, Z, norm, cmap, zlim, title, zlabel if c == 0 else "")
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cax = fig.add_axes([0.93, 0.9 - (r + 1) * (0.88 / rows) + 0.06, 0.012, 0.33])
        fig.colorbar(sm, cax=cax, label=zlabel + (" (shared in row)"))
    fig.suptitle(suptitle, y=0.995, fontsize=9)
    fig.subplots_adjust(left=0.0, right=0.91, top=0.9, bottom=0.02, wspace=0.02, hspace=0.12)
    save(fig, name)


def fig_A(t):
    ms = (("plain", "plain"), ("resolution_max_b1", "resolution-only"))
    for centered in (False, True):
        panels = {}
        for r, s in enumerate(("train_probe", "test_probe")):
            for c, (m, lab) in enumerate(ms):
                k = "A2d__%s__seed0" % m
                a, b, Z = grid_array(t[k], s)
                c0 = center_of(t[k], s)
                panels[(r, c)] = (a, b, Z - c0 if centered else Z,
                                  "%s, epoch 30, seed 0\n%s, centre CE %.3f" % (lab, SPLIT[s], c0))
        grid_figure(panels, 2, 2, "A_surface3d_%s" % ("centered" if centered else "absolute"),
                    "Analysis A: final solutions, %s CE. Target state (r=32, no filter), %s,\n"
                    "direction pair 0. Matched relative filter-wise perturbations: the same random "
                    "draws scaled by each\nsolution's own filter norms, not an identical physical plane. "
                    "Floors: contours of the same grid." % ("centre-subtracted" if centered else "absolute", BN),
                    centered, (8.6, 8.4))


def fig_B(t):
    ss = (16, 24, 32)
    for centered in (False, True):
        panels = {}
        for r, s in enumerate(("train_probe", "test_probe")):
            for c, res in enumerate(ss):
                k = "B2d__resolution_max_b1__seed0__ep06__r%d" % res
                a, b, Z = grid_array(t[k], s)
                c0 = center_of(t[k], s)
                panels[(r, c)] = (a, b, Z - c0 if centered else Z,
                                  "evaluated at r=%d%s\n%s, centre CE %.3f" % (
                                      res, " (target)" if res == 32 else "", SPLIT[s], c0))
        grid_figure(panels, 2, 3, "B_surface3d_%s" % ("centered" if centered else "absolute"),
                    "Analysis B: resolution-only checkpoint after epoch 6 (last updates at r=16), seed 0, "
                    "%s CE, %s per state.\nWeights, centre, directions (pair 0), subsets and axes are "
                    "identical across states; only the block-1 reduction changes.\nCounterfactual "
                    "evaluations of one set of weights, not three trained models."
                    % ("centre-subtracted" if centered else "absolute", BN),
                    centered, (11.5, 8.2))


def fig_C(t):
    pl = torch.load(INPUTS / "pca_plane.pt", map_location="cpu", weights_only=False)
    xs, ys = np.array(pl["grid"]["x"]), np.array(pl["grid"]["y"])
    co = np.array(pl["coords"])
    evr = pl["explained_variance_ratio"]
    fig = plt.figure(figsize=(11, 5.6))
    for i, s in enumerate(("train_probe", "test_probe")):
        Z = np.full((len(ys), len(xs)), np.nan)
        for r in t["C_plane"]:
            Z[r["j"], r["i"]] = r["splits"][s]["ce"]
        lo, hi = Z.min(), Z.max()
        zlim = (lo - 0.35 * (hi - lo), hi)
        norm = colors.Normalize(lo, hi)
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        X, Y = np.meshgrid(xs, ys)
        fc = cm.viridis(norm(Z))
        fc[..., 3] = 0.55          # translucent so the floor projection stays visible
        ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                        linewidth=0.15, edgecolor=(0, 0, 0, 0.2), shade=False)
        ax.contour(X, Y, Z, levels=10, zdir="z", offset=zlim[0], cmap=cm.viridis, norm=norm,
                   linewidths=0.6)
        for m, colr, lab in (("plain", "k", "plain"), ("resolution_max_b1", "#d62728", "resolution-only")):
            idx = [k for k, p in enumerate(pl["points"]) if p["method"] == m]
            ax.plot(co[idx, 0], co[idx, 1], zs=zlim[0], zdir="z", color=colr, lw=1.3, marker="o",
                    ms=3.5, label="%s: projected checkpoints (floor only)" % lab)
            for k in idx:
                ax.text(co[k, 0], co[k, 1], zlim[0], " %d" % pl["points"][k]["epochs_completed"],
                        color=colr, fontsize=6.5)
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1]); ax.set_zlim(*zlim)
        ax.set_box_aspect(BOX); ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_zticks([v for v in np.arange(0.5, hi + 1e-9, 0.5) if v >= lo - 1e-9])
        ax.set_xlabel("PC1 coordinate (%.0f%% var.)" % (100 * evr[0]))
        ax.set_ylabel("PC2 coordinate (%.0f%% var.)" % (100 * evr[1]))
        ax.set_zlabel("cross-entropy")
        ax.tick_params(pad=0, labelsize=6.5)
        ax.set_title("%s, target state, %s" % (SPLIT[s], BN), pad=0)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis), ax=ax, shrink=0.55, pad=0.08,
                     label="CE of reconstructed in-plane points")
        if i == 0:
            ax.legend(loc="upper left", fontsize=6.5, frameon=False)
    fig.suptitle("Analysis C: target objective on the shared PCA plane of the seed-0 plain and "
                 "resolution-only checkpoints (epochs 0, 6, 12, 18, 30).\nSurface = measured 25x25 grid of "
                 "reconstructed in-plane points. Checkpoints are drawn only on the floor at their "
                 "projected coordinates: they lie off\nthe plane (relative residual up to 0.66 at epoch 6) "
                 "and their own CE is not this surface (see C_checkpoint_ce). PC1+PC2 = 91% of the variance "
                 "of 10 points.", fontsize=8.5)
    fig.subplots_adjust(left=0.0, right=0.97, top=0.95, bottom=0.02, wspace=0.08)
    save(fig, "C_surface3d_plane")


def fig_C_checkpoints(t):
    by = {(r["file"], r["bn_policy"], r["state_label"]): r for r in t["C_checkpoints"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, constrained_layout=True)
    ep = (0, 6, 12, 18, 30)
    for ax, s in zip(axes, ("train_probe", "test_probe")):
        for m, colr, lab in (("plain", "k", "plain"), ("resolution_max_b1", "#d62728", "res.-only")):
            f = lambda e: "%s__seed0__ep%02d.pt" % (m, e)
            rt = [by[(f(e), "recalibrated", "target")]["splits"][s]["ce"] for e in ep]
            ax.plot(ep, rt, "-o", color=colr, label="%s: recalibrated BN, target state" % lab)
            st = [by[(f(e), "saved_stats", "target")]["splits"][s]["ce"] for e in ep]
            ax.plot(ep, st, ":s", color=colr, mfc="none", label="%s: saved BN stats, target state" % lab)
            if m == "resolution_max_b1":
                ru = [by[(f(e), "recalibrated", "used")]["splits"][s]["ce"] for e in (6, 12)]
                su = [by[(f(e), "saved_stats", "used")]["splits"][s]["ce"] for e in (6, 12)]
                ax.plot((6, 12), ru, "^", color="#1f77b4", ms=7,
                        label="res.-only: recalibrated BN, own state (r=16 / r=24)")
                ax.plot((6, 12), su, "v", color="#1f77b4", mfc="none", ms=7,
                        label="res.-only: saved BN stats, own state (r=16 / r=24)")
        ax.set_yscale("log")
        ax.set_xticks(ep)
        ax.set_xlabel("epochs completed")
        ax.set_title("actual checkpoint CE, seed 0, %s" % SPLIT[s])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cross-entropy (log scale)")
    axes[1].legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Analysis C companion: CE of the actual (off-plane) checkpoints, by BatchNorm convention. "
                 "Epoch 0 is the shared initialisation;\nfor plain the own state equals the target state. "
                 "Only checkpoints were sampled; lines join them and are not observed paths.")
    save(fig, "C_checkpoint_ce")


def interactive(t):
    import plotly.graph_objects as go
    import plotly.io as pio
    parts, first = [], True

    def add(title, grids, zlab, centered):
        nonlocal first
        lo = min(g[2].min() for g in grids); hi = max(g[2].max() for g in grids)
        m = max(abs(lo), abs(hi))
        cmin, cmax, cs = ((-m, m, "RdBu_r") if centered else (lo, hi, "Viridis"))
        for a, b, Z, sub in grids:
            f = go.Figure(go.Surface(x=a, y=b, z=Z, cmin=cmin, cmax=cmax, colorscale=cs,
                                     contours={"z": {"show": True, "project": {"z": True}}}))
            f.update_layout(title="%s<br><sub>%s</sub>" % (title, sub), height=520,
                            scene=dict(xaxis_title="a / PC1", yaxis_title="b / PC2", zaxis_title=zlab,
                                       zaxis=dict(range=[lo, hi]), aspectratio=dict(x=1, y=1, z=0.75),
                                       camera=dict(eye=dict(x=-1.25, y=-1.6, z=1.0))),
                            margin=dict(l=0, r=0, t=60, b=0))
            parts.append(pio.to_html(f, full_html=False, include_plotlyjs="inline" if first else False))
            first = False

    for s in ("train_probe", "test_probe"):
        for centered in (False, True):
            g = []
            for m, lab in (("plain", "plain"), ("resolution_max_b1", "resolution-only")):
                a, b, Z = grid_array(t["A2d__%s__seed0" % m], s)
                c0 = center_of(t["A2d__%s__seed0" % m], s)
                g.append((a, b, Z - c0 if centered else Z, "%s, epoch 30, seed 0, %s" % (lab, SPLIT[s])))
            add("A: final solutions (matched relative perturbations, %s)" % BN, g,
                "CE - centre" if centered else "CE", centered)
            g = []
            for res in (16, 24, 32):
                k = "B2d__resolution_max_b1__seed0__ep06__r%d" % res
                a, b, Z = grid_array(t[k], s)
                c0 = center_of(t[k], s)
                g.append((a, b, Z - c0 if centered else Z, "res.-only epoch 6, seed 0, r=%d, %s" % (res, SPLIT[s])))
            add("B: fixed weights across states (%s)" % BN, g, "CE - centre" if centered else "CE", centered)
    html = ("<html><head><meta charset='utf-8'><title>landscape_v1 surfaces</title></head><body>"
            "<h2>landscape_v1: measured grids rendered as surfaces</h2><p>Plain vs resolution-only, "
            "resbench checkpoints (not the unified runs). Shared colour and z range within each "
            "comparison block. Gaussian-only and combined: no checkpoints, not shown.</p>"
            + "".join("<div style='display:inline-block;width:49%%'>%s</div>" % p for p in parts)
            + "</body></html>")
    (FIG / "surfaces_interactive.html").write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "studies" / "landscape_v1" / "results"))
    t = load(Path(ap.parse_args().results))
    fig_A(t); fig_B(t); fig_C(t); fig_C_checkpoints(t); interactive(t)
    print("surfaces written to", FIG)


if __name__ == "__main__":
    main()

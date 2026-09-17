"""Paper figures of the geometry completion (PDF + PNG), from measured values only.

    py -m geometry_final.figures

No smoothing, no fitted surfaces, no clipping: every surface joins measured 41x41
vertices; a zoom panel is the subset of the same vertices within |a|, |b| <= 0.1.
Colours and names follow the manuscript (paper/tools/build_assets.py).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from . import common as C  # noqa: E402

FIG = C.STUDY / "figures"
TAB = C.STUDY / "tables"
M = list(C.METHODS)
LONG = dict(zip(M, ["Plain", "Resolution (R)", "Gaussian (G)", "Combined (RG)"]))
SHORTN = dict(zip(M, ["Plain", "R", "G", "RG"]))
COLOR = dict(zip(M, ["#333333", "#1976b2", "#dc7b13", "#8653aa"]))
POL = {"centre_frozen": "Centre-frozen BN", "pointwise": "Pointwise-recalibrated BN", "saved": "Saved BN"}
SPLIT = {"train_probe": "training probe", "test_probe": "test probe"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
                     "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
ELEV, AZIM = 25, -55
CAPTIONS = {}


def save(fig, name, caption, sources):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(FIG / (name + ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    CAPTIONS[name] = {"caption": caption, "sources": sources}


def _matrix(df, m, pol, col):
    g = df[(df.method == m) & (df.bn_policy == pol)]
    z = g.pivot(index="b", columns="a", values=col).sort_index().sort_index(axis=1)
    return z


def _surface(ax, z, norm, zlim, title, ticks):
    xx, yy = np.meshgrid(z.columns.to_numpy(), z.index.to_numpy())
    Z = z.to_numpy()
    ax.plot_surface(xx, yy, Z, cmap="viridis", norm=norm, rstride=1, cstride=1, linewidth=.16,
                    edgecolor=(0, 0, 0, .16), antialiased=True, shade=False)
    lim = float(np.abs(z.columns).max())
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(*zlim)
    ax.set_xticks([-lim, 0, lim]); ax.set_yticks([0, lim])
    if ticks is not None:
        ax.set_zticks(ticks)
    ax.set_xlabel("$a$", fontsize=8, labelpad=-7); ax.set_ylabel("$b$", fontsize=8, labelpad=-7)
    ax.tick_params(labelsize=6.5, pad=-3)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_title(title, fontsize=9, pad=4)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis._axinfo["grid"]["linewidth"] = .35
        axis._axinfo["grid"]["color"] = (.75, .75, .75, .5)


def _nice_ticks(lo, hi, n=4):
    step = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(step))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= step), default=step)
    return np.arange(np.ceil(lo / step) * step, hi + 1e-12, step)


def surfaces_row(df, pol, split, name, zoom=None, caption_extra=""):
    col = split + "_ce_centred"
    d = df if zoom is None else df[(df.a.abs() <= zoom + 1e-9) & (df.b.abs() <= zoom + 1e-9)]
    mats = [_matrix(d, m, pol, col) for m in M]
    for z in mats:
        assert not z.isna().any().any(), "missing vertex"
    lo = min(float(z.to_numpy().min()) for z in mats)
    hi = max(float(z.to_numpy().max()) for z in mats)
    norm = Normalize(lo, hi)
    fig = plt.figure(figsize=(7.0, 2.6))
    for i, (m, z) in enumerate(zip(M, mats)):
        ax = fig.add_axes([.012 + .245 * i, .15, .235, .77], projection="3d")
        _surface(ax, z, norm, (lo, hi), LONG[m], _nice_ticks(max(lo, 0), hi, 2))
    cax = fig.add_axes([.30, .115, .40, .04])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax, orientation="horizontal",
                      ticks=_nice_ticks(max(lo, 0), hi, 4))
    cb.set_label("Height and colour: CE above centre (nats)" + ("" if zoom is None else ", zoom |a|,|b| ≤ %g" % zoom),
                 fontsize=8, labelpad=1)
    cb.ax.tick_params(labelsize=7, pad=1, length=2)
    n = len(mats[0])
    grid_txt = "the %d-by-%d measured vertices" % (n, n)
    cap = ("Loss slices of the four final seed-0 models under %s: CIFAR-10 / ResNet-20-BN / SGD, 30 epochs, "
           "1k-image %s. Surfaces join %s of $L(\\theta_0+a d_1+b d_2)-L(\\theta_0)$ over %s; no smoothing or "
           "fitting. Viewing angle, axis limits and colour scale are identical across panels (range %.3g to %.3g nats). "
           "The random draws are shared, but each model scales them by its own filter norms, so the four panels "
           "do not show one physical plane. One seed and one direction pair: an illustration, not a ranking.%s"
           % (POL[pol].replace("BN", "BatchNorm").lower() if pol != "centre_frozen" else
              "centre-frozen BatchNorm (statistics recalibrated once at the unperturbed weights on 2,000 training images, then held fixed)",
              SPLIT[split], grid_txt, "$[-0.5,0.5]^2$" if zoom is None else "the zoom $[-%g,%g]^2$ (a subset of the same 41-by-41 grid)" % (zoom, zoom),
              lo, hi, caption_extra))
    save(fig, name, cap, ["tables/T_surface_vertices_seed0.csv"])


def policy_comparison(df, split, name, shared=True):
    col = split + "_ce_centred"
    pols = ("pointwise", "centre_frozen")
    mats = {(p, m): _matrix(df, m, p, col) for p in pols for m in M}
    rng = {p: (min(float(mats[(p, m)].to_numpy().min()) for m in M), max(float(mats[(p, m)].to_numpy().max()) for m in M))
           for p in pols}
    if shared:
        lo, hi = min(r[0] for r in rng.values()), max(r[1] for r in rng.values())
        rng = {p: (lo, hi) for p in pols}
    fig = plt.figure(figsize=(7.0, 5.8))
    for r_, p in enumerate(pols):
        norm = Normalize(*rng[p])
        for i, m in enumerate(M):
            ax = fig.add_axes([.012 + .245 * i, .61 - .5 * r_, .235, .33], projection="3d")
            _surface(ax, mats[(p, m)], norm, rng[p], "%s\n%s" % (LONG[m], POL[p]), _nice_ticks(max(rng[p][0], 0), rng[p][1], 2))
        if not shared or r_ == 1:
            cax = fig.add_axes([.30, (.565 - .5 * r_) if not shared else .065, .40, .016])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax, orientation="horizontal",
                              ticks=_nice_ticks(max(rng[p][0], 0), rng[p][1], 4))
            cb.set_label("CE above centre (nats)" + ("" if shared else ", %s row" % POL[p]), fontsize=8, labelpad=1)
            cb.ax.tick_params(labelsize=7, pad=1, length=2)
    cap = ("Same seed-0 checkpoints, direction pair, filter scaling, 41-by-41 grid over $[-0.5,0.5]^2$ and 1k-image %s; "
           "only the BatchNorm evaluation policy differs between rows (top: recalibrated at every vertex; bottom: "
           "recalibrated once at the centre, then frozen). Centre losses are identical under the two policies. %s "
           "Heights are CE above each model's own centre; no smoothing or fitting."
           % (SPLIT[split], "Both rows share one colour scale and z-range (%.3g to %.3g nats)." % rng["pointwise"] if shared else
              "Each row has its own common scale (pointwise %.3g to %.3g; centre-frozen %.3g to %.3g nats); panels are "
              "comparable within a row, not across rows." % (*rng["pointwise"], *rng["centre_frozen"])))
    save(fig, name, cap, ["tables/T_surface_vertices_seed0.csv"])


def axis_profiles(df, name):
    """Measured CE above centre along the two grid axes, both policies, log scale."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), layout="constrained", sharex=True)
    for c_, split in enumerate(("train_probe", "test_probe")):
        for r_, (axname, fixed) in enumerate((("a", "b"), ("b", "a"))):
            ax = axes[r_, c_]
            for m in M:
                for pol, ls in (("centre_frozen", "-"), ("pointwise", "--")):
                    g = df[(df.method == m) & (df.bn_policy == pol) & (df[fixed] == 0)].sort_values(axname)
                    y = g[split + "_ce_centred"].to_numpy()
                    x = g[axname].to_numpy()
                    pos = y > 0
                    ax.plot(x[pos], y[pos], ls, color=COLOR[m], lw=1.1, marker="o", ms=1.6)
            ax.set_yscale("log")
            ax.set_title("%s, along $%s$ ($%s=0$)" % (SPLIT[split].capitalize(), axname, fixed), fontsize=9)
            ax.grid(alpha=.18)
            if c_ == 0:
                ax.set_ylabel("CE above centre (nats)")
            if r_ == 1:
                ax.set_xlabel("Coefficient")
    h = [plt.Line2D([], [], color=COLOR[m], lw=1.2, label=LONG[m]) for m in M]
    h += [plt.Line2D([], [], color=".3", ls="-", label="centre-frozen"), plt.Line2D([], [], color=".3", ls="--", label="pointwise")]
    fig.legend(handles=h, loc="outside lower center", ncol=6, frameon=False, fontsize=7.5)
    n_nonpos = int((df[(df.a == 0) | (df.b == 0)][["train_probe_ce_centred", "test_probe_ce_centred"]] <= 0).sum().sum())
    cap = ("Measured CE above centre along the two axes of the seed-0 41-by-41 grid (each axis is one filter-normalised "
           "random direction), centre-frozen (solid) and pointwise-recalibrated (dashed) BatchNorm, CIFAR-10 / "
           "ResNet-20-BN / SGD final checkpoints, 1k-image probes. Logarithmic vertical axis; the centre and %d "
           "non-positive values (including the centre itself) cannot be drawn on it and are omitted from the lines only."
           % n_nonpos)
    save(fig, name, cap, ["tables/T_surface_vertices_seed0.csv"])


def sensitivity(name, policies, caption_policy):
    s = pd.read_csv(C.V3_TABLES / "A_sensitivity_per_seed.csv")
    fig, axes = plt.subplots(1, 2 * len(policies), figsize=(3.5 * len(policies), 2.35), layout="constrained")
    k = 0
    rows = []
    for pol in policies:
        for split in ("train_probe", "test_probe"):
            ax = axes[k]; k += 1
            ax.axhline(0, color=".35", lw=.7)
            base = s[(s.policy == pol) & (s.split == split) & (s.method == "plain")].set_index(["seed", "amplitude"]).S_mean
            for m in M[1:]:
                g = s[(s.policy == pol) & (s.split == split) & (s.method == m)].copy()
                g["delta"] = g.S_mean.values - base.loc[list(zip(g.seed, g.amplitude))].values
                agg = g.groupby("amplitude").delta.agg(["mean", "std", "size", lambda v: int((v < 0).sum())]).reset_index()
                agg.columns = ["amplitude", "mean", "sd", "n", "n_below"]
                assert (agg.n == 5).all() and len(agg) == 10
                for _, r in agg.iterrows():
                    rows.append({"policy": pol, "split": split, "method": m, "amplitude": r.amplitude, "mean_delta": r["mean"],
                                 "sd_delta": r.sd, "n_seeds": int(r.n), "n_seeds_below_plain": int(r.n_below)})
                x, y, sd = agg.amplitude.to_numpy(), agg["mean"].to_numpy(), agg.sd.to_numpy()
                ax.plot(x, y, "o-", color=COLOR[m], ms=2.7, lw=1.2)
                ax.fill_between(x, y - sd, y + sd, color=COLOR[m], alpha=.14, lw=0)
            ax.set_xscale("log")
            ax.set_xticks([.005, .025, .1, .5], [".005", ".025", ".1", ".5"])
            ax.set_xlabel(r"Amplitude $\varepsilon$", fontsize=8)
            ax.set_title({"saved": "Saved", "recalibrated": "Pointwise", "centre_frozen": "Centre-frozen"}[pol] + " / "
                         + ("train" if split == "train_probe" else "test"), fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=.18)
    axes[0].set_ylabel(r"$\Delta S$ (nats)", fontsize=9)
    h = [plt.Line2D([], [], color=COLOR[m], marker="o", ms=3, lw=1.2, label=LONG[m]) for m in M[1:]]
    fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False)
    cap = ("Finite sensitivity relative to Plain, %s: CIFAR-10 / ResNet-20-BN / SGD, final 30-epoch checkpoints, "
           "1k-image probes. For each seed, $S$ averages 20 filter-normalised directions (both signs) and "
           "$\\Delta S=S_{\\rm method}-S_{\\rm Plain}$; points are the mean of five seed-level differences and bands "
           "$\\pm1$ sample SD across seeds. Negative means a smaller CE increase. Logarithmic amplitude axis; vertical "
           "scales differ between panels. Existing measurements; no new evaluation." % caption_policy)
    save(fig, name, cap, ["landscape_v3 tables/A_sensitivity_per_seed.csv", "tables/T_sensitivity_paired.csv"])
    return pd.DataFrame(rows)


def hessian_vs_random(name):
    """Symmetric rise along the leading eigendirection vs the 20 random directions, same r units."""
    cuts = pd.read_csv(C.V3_TABLES / "D_hessian_cuts.csv")
    sens = pd.read_csv(C.V3_TABLES / "A_sensitivity_per_direction.csv")
    x3 = pd.read_csv(TAB / "X3_direction_curvature_per_checkpoint.csv")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5), layout="constrained", sharey=True)
    for ax, m in zip(axes, M):
        c = cuts[(cuts.method == m) & (cuts.seed == 0) & (cuts.source_probe == "train_probe") & (cuts.coords == "relative")
                 & (cuts.source_policy == "centre_frozen") & (cuts.vector == "top1")]
        for pol, ls, lab in (("centre_frozen", "-", "top eigendirection, centre-frozen"), ("recalibrated", "--", "same direction, pointwise")):
            g = c[c.eval_policy == pol].set_index("t").train_probe_ce
            ts = np.array(sorted(t for t in g.index if t > 0))
            S = 0.5 * (g.loc[ts].to_numpy() + g.loc[-ts].to_numpy()) - g.loc[0.0]
            ok = S > 0
            ax.plot(ts[ok], S[ok], ls, color="#b2182b", lw=1.1, marker="o", ms=2)
        for pol, ls in (("centre_frozen", "-"), ("recalibrated", "--")):
            g = sens[(sens.method == m) & (sens.seed == 0) & (sens.split == "train_probe") & (sens.policy == pol)]
            agg = g.groupby("amplitude").S.agg(["mean", "min", "max"]).reset_index()
            ax.plot(agg.amplitude, agg["mean"], ls, color=COLOR[m], lw=1.1, marker="o", ms=2)
            if pol == "centre_frozen":
                ax.fill_between(agg.amplitude, agg["min"].clip(lower=1e-9), agg["max"], color=COLOR[m], alpha=.15, lw=0)
        r = x3[(x3.method == m) & (x3.seed == 0) & (x3.probe == "train_probe") & (x3.policy == "centre_frozen")].iloc[0]
        tt = np.geomspace(5e-4, 0.5, 50)
        ax.plot(tt, 0.5 * r.r_units_extreme_top1 * tt ** 2, ":", color="#b2182b", lw=.8)
        ax.plot(tt, 0.5 * r.r_units_random_mean_dHd * tt ** 2, ":", color=COLOR[m], lw=.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(1e-6, 1e13)
        ax.set_title(LONG[m], fontsize=9)
        ax.set_xlabel("RMS relative displacement $r$", fontsize=8)
        ax.grid(alpha=.18)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Symmetric rise $S$ (nats)")
    h = [plt.Line2D([], [], color="#b2182b", lw=1.1, label="leading eigendirection"),
         plt.Line2D([], [], color=".45", lw=1.1, label="20 random directions (mean; band min–max)"),
         plt.Line2D([], [], color=".3", ls="-", label="centre-frozen"), plt.Line2D([], [], color=".3", ls="--", label="pointwise"),
         plt.Line2D([], [], color=".3", ls=":", label="local quadratic ½·curvature·$r^2$")]
    fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False, fontsize=7.5)
    cap = ("Measured symmetric rise $S(r)=\\tfrac12[L(\\theta+r\\delta)+L(\\theta-r\\delta)]-L(\\theta)$ on the 1k-image training probe for "
           "the seed-0 final models, with displacement expressed as RMS relative block displacement $r$ (so random and "
           "Hessian directions share one axis). Red: the leading eigenvector of the block-relative centre-frozen Hessian "
           "(each model's own; not a shared direction), scaled to $r=1$. Colour: 20 filter-normalised random directions "
           "(mean, band min--max over directions, centre-frozen). Solid: centre-frozen BatchNorm; dashed: the same "
           "perturbations with pointwise recalibration. Dotted: $\\tfrac12 G\\lambda_{\\rm rel}r^2$ and "
           "$\\tfrac12\\overline{d^\\top Hd}\\,r^2$ from the HVP measurements. Log--log axes; non-positive $S$ omitted. "
           "Existing measurements only.")
    save(fig, name, cap, ["landscape_v3 tables/D_hessian_cuts.csv", "landscape_v3 tables/A_sensitivity_per_direction.csv",
                          "tables/X3_direction_curvature_per_checkpoint.csv"])


def hessian_planes(name):
    import glob
    rows = []
    for f in glob.glob(str(C.V3_RAW / "*" / "v3" / "eval" / "hplane__*_train_centre_frozen_rel__top1_top2.jsonl")):
        for line in open(f):
            r = json.loads(line)
            rows.append({"method": [m for m in M if C.SHORT[m] == r["problem"].split("_s0")[0]][0], "u1": r["u1"], "u2": r["u2"],
                         "t1": r["t1"], "t2": r["t2"], "T1": r["T1"], "T2": r["T2"],
                         "train_probe_ce": r["splits"]["train_probe"]["ce"], "test_probe_ce": r["splits"]["test_probe"]["ce"]})
    df = pd.DataFrame(rows)
    for sp in ("train_probe", "test_probe"):
        c = df[(df.u1 == 0) & (df.u2 == 0)].set_index("method")[sp + "_ce"]
        df[sp + "_ce_centred"] = df[sp + "_ce"] - c.loc[df.method].to_numpy()
    df.to_csv(TAB / "T_hessian_plane_vertices_seed0_centre_frozen.csv", index=False)
    mats = []
    for m in M:
        g = df[df.method == m]
        z = g.pivot(index="u2", columns="u1", values="train_probe_ce_centred").sort_index().sort_index(axis=1)
        assert z.shape == (41, 41) and not z.isna().any().any()
        mats.append(z)
    lo = min(float(z.to_numpy().min()) for z in mats); hi = max(float(z.to_numpy().max()) for z in mats)
    norm = Normalize(lo, hi)
    fig = plt.figure(figsize=(7.0, 2.6))
    for i, (m, z) in enumerate(zip(M, mats)):
        ax = fig.add_axes([.012 + .245 * i, .15, .235, .77], projection="3d")
        _surface(ax, z, norm, (lo, hi), LONG[m], _nice_ticks(max(lo, 0), hi, 2))
        ax.set_xlabel("$u_1$", fontsize=8, labelpad=-7); ax.set_ylabel("$u_2$", fontsize=8, labelpad=-7)
    cax = fig.add_axes([.30, .115, .40, .04])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax, orientation="horizontal",
                      ticks=_nice_ticks(max(lo, 0), hi, 4))
    cb.set_label("Height and colour: CE above centre (nats)", fontsize=8, labelpad=1)
    cb.ax.tick_params(labelsize=7, pad=1, length=2)
    T1, T2 = float(df.T1.iloc[0]), float(df.T2.iloc[0])
    cap = ("Existing seed-0 planes spanned by each model's own two leading eigenvectors of the centre-frozen, block-relative "
           "Hessian (1k-image training probe), mapped to weight space ($\\delta=Aq$) and scaled to $r=1$; the eigenvectors "
           "differ between models and are not orthogonal in weight space. Vertices $u_1,u_2\\in[-1,1]$ (41-by-41) are displaced "
           "by $t_i=u_iT_i$ with common extents $T_1=%.2g$ and $T_2=%.2g$ in $r$ units, fixed from Plain's eigenvalues so "
           "that Plain's quadratic model rises by 2 nats at each axis end. These extents are about 20 times smaller than one step "
           "of the random grid (0.025). Heights are measured CE above centre; identical camera, limits and colour scale "
           "(%.3g to %.3g nats)." % (T1, T2, lo, hi))
    save(fig, name, cap, ["landscape_v3 raw hplane__*_train_centre_frozen_rel__top1_top2.jsonl",
                          "tables/T_hessian_plane_vertices_seed0_centre_frozen.csv"])


def main(with_new=True):
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    main_sens = sensitivity("F_sensitivity_centre_frozen_pointwise", ["centre_frozen", "recalibrated"],
                            "centre-frozen (left pair) and pointwise-recalibrated (right pair) BatchNorm")
    app_sens = sensitivity("F_sensitivity_saved_appendix", ["saved"], "saved BatchNorm statistics")
    pd.concat([main_sens, app_sens]).to_csv(TAB / "T_sensitivity_paired.csv", index=False)
    hessian_vs_random("F_hessian_vs_random_rise_seed0")
    hessian_planes("F_hessian_plane_top1_top2_centre_frozen_seed0")
    if with_new:
        from .data import grid_vertices
        df = grid_vertices()
        df.to_csv(TAB / "T_surface_vertices_seed0.csv", index=False)
        for split in ("train_probe", "test_probe"):
            tag = "" if split == "train_probe" else "_test"
            surfaces_row(df, "centre_frozen", split, "F_surfaces_centre_frozen_seed0" + tag)
            surfaces_row(df, "centre_frozen", split, "F_surfaces_centre_frozen_seed0_zoom" + tag, zoom=0.1,
                         caption_extra=" Zoom of the full-range figure: the same measured vertices restricted to $|a|,|b|\\le0.1$ (9-by-9), with a common scale of its own.")
            surfaces_row(df, "pointwise", split, "F_surfaces_pointwise_seed0" + tag)
            policy_comparison(df, split, "F_surfaces_policy_comparison_shared_scale" + tag, shared=True)
            policy_comparison(df, split, "F_surfaces_policy_comparison_rowwise_scale" + tag, shared=False)
            surfaces_row(df, "pointwise", split, "F_surfaces_pointwise_seed0_zoom" + tag, zoom=0.1,
                         caption_extra=" Zoom of the full-range figure: the same measured vertices restricted to $|a|,|b|\\le0.1$ (9-by-9), with a common scale of its own.")
        axis_profiles(df, "F_surface_axis_profiles_seed0")
    old = json.loads((FIG / "CAPTIONS.json").read_text()) if (FIG / "CAPTIONS.json").exists() else {}
    old.update(CAPTIONS)
    (FIG / "CAPTIONS.json").write_text(json.dumps(old, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    main(with_new="--existing-only" not in sys.argv)

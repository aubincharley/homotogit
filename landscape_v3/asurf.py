"""Surfaces: wide random planes (all seeds, seed-0 41x41) and seed-0 Hessian planes.

Every vertex is a measured point; missing or non-finite vertices are drawn as
grey crosses on the floor and counted in the panel title.
"""
from __future__ import annotations

import json

import numpy as np
from matplotlib import cm, colors

from . import common as C
from .acommon import AZIM, BOX, ELEV, FIG, LAB, SPL, plt, save, write_csv


def grid_values(D, m, s, spec_fn, grid, policy, split):
    tgt = C.target(m)
    n = len(grid)
    Z = np.full((n, n), np.nan)
    for i, a in enumerate(grid):
        for j, b in enumerate(grid):
            ce = D.ce(C.pkey(m, s, C.FINAL, tgt, policy, "probes", spec_fn(a, b)), split)
            if ce is not None:
                Z[i, j] = ce
    return Z


def centre_of(D, m, s, policy, split, Z, grid):
    c = D.ce(C.pkey(m, s, C.FINAL, C.target(m), policy, "probes", "c"), split)
    if c is None:
        i0 = int(np.argmin(np.abs(np.asarray(grid))))
        c = Z[i0, i0]
    return c


def panel3d(ax, g, Z, norm, cmap, zlim, title, zlabel, g2=None):
    A, B = np.meshgrid(g, g if g2 is None else g2, indexing="ij")
    Zm = np.ma.masked_invalid(Z)
    ax.plot_surface(A, B, Zm, facecolors=cmap(norm(np.nan_to_num(Z, nan=zlim[0]))), rstride=1, cstride=1,
                    linewidth=0.1, edgecolor="k", antialiased=True, shade=False)
    bad = ~np.isfinite(Z)
    if bad.any():
        ax.scatter(A[bad], B[bad], np.full(bad.sum(), zlim[0]), color="0.5", marker="x", s=8)
    ax.set_zlim(*zlim)
    ax.view_init(ELEV, AZIM)
    ax.set_box_aspect(BOX)
    ax.set_title(title + ("  [%d missing]" % bad.sum() if bad.any() else ""), fontsize=7.5)
    ax.set_zlabel(zlabel, fontsize=7)
    ax.tick_params(labelsize=6)


def surface_set(D, name, seeds, grid, spec_fn, policy, title, axis_names, splits=C.PROBES, prefix="E"):
    """Absolute + centre-subtracted 3D and contour figures with shared scales per split."""
    out_rows = []
    for split in splits:
        Zs, Cs = {}, {}
        for s in seeds:
            for m in C.METHODS:
                Z = grid_values(D, m, s, spec_fn, grid, policy, split)
                Zs[(m, s)] = Z
                Cs[(m, s)] = Z - centre_of(D, m, s, policy, split, Z, grid)
                out_rows.append({"figure": name, "method": m, "seed": s, "split": split, "n_vertices": Z.size,
                                 "n_missing": int((~np.isfinite(Z)).sum()), "min_ce": float(np.nanmin(Z)) if np.isfinite(Z).any() else np.nan,
                                 "max_ce": float(np.nanmax(Z)) if np.isfinite(Z).any() else np.nan,
                                 "min_centred": float(np.nanmin(Cs[(m, s)])) if np.isfinite(Z).any() else np.nan,
                                 "max_centred": float(np.nanmax(Cs[(m, s)])) if np.isfinite(Z).any() else np.nan})
        allZ = np.concatenate([z[np.isfinite(z)] for z in Zs.values()] or [np.array([0.0, 1.0])])
        allC = np.concatenate([z[np.isfinite(z)] for z in Cs.values()] or [np.array([-1.0, 1.0])])
        if allZ.size == 0:
            continue
        zl = (float(allZ.min()), float(allZ.max()))
        cmax = float(np.abs(allC).max()) if allC.size else 1.0
        cl = (float(min(allC.min(), 0.0)), float(allC.max()))
        normA = colors.LogNorm(vmin=max(zl[0], 1e-3), vmax=zl[1])
        normC = colors.TwoSlopeNorm(vmin=-max(abs(cl[0]), 1e-6), vcenter=0.0, vmax=max(cmax, 1e-6))
        for kind, ZZ, norm, cmap, zlim, zlab in (("absolute", Zs, normA, cm.viridis, zl, "CE"),
                                                 ("centred", Cs, normC, cm.RdBu_r, cl, "CE − centre CE")):
            n_rows = len(seeds)
            fig = plt.figure(figsize=(13, 3.3 * n_rows))
            for r_, s in enumerate(seeds):
                for c_, m in enumerate(C.METHODS):
                    ax = fig.add_subplot(n_rows, 4, r_ * 4 + c_ + 1, projection="3d")
                    panel3d(ax, grid, ZZ[(m, s)], norm, cmap, zlim, "%s, seed %d" % (LAB[m], s), zlab if c_ == 0 else "")
                    ax.set_xlabel(axis_names[0], fontsize=6.5)
                    ax.set_ylabel(axis_names[1], fontsize=6.5)
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            fig.colorbar(sm, ax=fig.axes, shrink=0.5, pad=0.02, label=zlab)
            fig.suptitle("%s — %s, %s" % (title, SPL[split], "absolute CE" if kind == "absolute" else "centre-subtracted CE"))
            save(fig, "%s_%s_surface3d_%s_%s" % (prefix, name, kind, split))
            fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3.0 * n_rows), constrained_layout=True, squeeze=False)
            for r_, s in enumerate(seeds):
                for c_, m in enumerate(C.METHODS):
                    ax = axes[r_, c_]
                    Z = ZZ[(m, s)]
                    levels = (np.geomspace(max(zlim[0], 1e-3), zlim[1], 14) if kind == "absolute"
                              else np.linspace(-max(abs(cl[0]), 1e-6), cmax, 15))
                    cs = ax.contourf(grid, grid, np.ma.masked_invalid(Z).T, levels=levels, cmap=cmap, norm=norm)
                    ax.contour(grid, grid, np.ma.masked_invalid(Z).T, levels=levels, colors="k", linewidths=0.25)
                    bad = ~np.isfinite(Z)
                    if bad.any():
                        A, B = np.meshgrid(grid, grid, indexing="ij")
                        ax.scatter(A[bad], B[bad], color="0.5", marker="x", s=8)
                    ax.plot(0, 0, "k+", ms=6)
                    ax.set_aspect("equal")
                    ax.set_title("%s, seed %d" % (LAB[m], s), fontsize=7.5)
                    ax.set_xlabel(axis_names[0], fontsize=7); ax.set_ylabel(axis_names[1], fontsize=7)
            fig.colorbar(cs, ax=axes, shrink=0.6, label=zlab)
            fig.suptitle("%s — %s, %s (contours)" % (title, SPL[split], "absolute CE" if kind == "absolute" else "centre-subtracted CE"))
            save(fig, "%s_%s_contour_%s_%s" % (prefix, name, kind, split))
    return out_rows


def random_planes(D, summary):
    rows = []
    rows += surface_set(D, "wide_random_seed0_g41", [0], C.WIDE41, lambda a, b: C.spec_r2(0, 1, a, b), C.POINTWISE,
                        "Random directions 0/1, [−0.5, 0.5]², 41×41, recalibrated at every point, final state",
                        ("ε along direction 0", "ε along direction 1"))
    rows += surface_set(D, "wide_random_allseeds_g21", list(C.SEEDS), C.WIDE21, lambda a, b: C.spec_r2(0, 1, a, b),
                        C.POINTWISE, "Random directions 0/1, [−0.5, 0.5]², 21×21, recalibrated at every point",
                        ("ε, dir 0", "ε, dir 1"))
    write_csv("E_random_planes", rows)
    summary["E_random_planes"] = rows


def hessian_planes(D, summary):
    rows = []
    for pol in C.HESS_POLICIES:
        for v1, v2 in (("top1", "top2"), ("top1", "min")):
            rows += hessian_plane_set(D, pol, v1, v2, "hessian_plane_%s_%s_%s_seed0" % (pol, v1, v2))
    write_csv("E_hessian_planes", rows)
    summary["E_hessian_planes"] = rows


def hessian_plane_set(D, pol, v1, v2, name):
    """Like surface_set, but the spec (problem id) differs per method."""
    s = C.PRIMARY_SEED
    ugrid = list(C.HESS_PLANE_U)
    rows = []
    ext = None
    for m in C.METHODS:
        r = D.rows.get(C.pkey(m, s, C.FINAL, C.target(m), pol, "probes",
                              C.spec_h2(C.hess_problem(m, s, "train_probe", pol, "relative"), v1, v2, 1.0, 1.0)))
        if r is not None:
            ext = (r["T1"], r["T2"])
            break
    if ext is None:
        return rows
    grid = [u * ext[0] for u in ugrid]
    grid2 = [u * ext[1] for u in ugrid]
    for split in C.PROBES:
        Zs, Cs = {}, {}
        for m in C.METHODS:
            pid = C.hess_problem(m, s, "train_probe", pol, "relative")
            Z = grid_values(D, m, s, lambda a, b, pid=pid: C.spec_h2(pid, v1, v2, a, b), ugrid, pol, split)
            Zs[m] = Z
            Cs[m] = Z - centre_of(D, m, s, pol, split, Z, ugrid)
            rows.append({"figure": name, "method": m, "seed": s, "split": split, "T1": ext[0], "T2": ext[1],
                         "n_missing": int((~np.isfinite(Z)).sum()),
                         "min_ce": float(np.nanmin(Z)) if np.isfinite(Z).any() else np.nan,
                         "max_ce": float(np.nanmax(Z)) if np.isfinite(Z).any() else np.nan})
        good = [z[np.isfinite(z)] for z in Zs.values() if np.isfinite(z).any()]
        if not good:
            continue
        allZ, allC = np.concatenate(good), np.concatenate([z[np.isfinite(z)] for z in Cs.values() if np.isfinite(z).any()])
        zl = (float(allZ.min()), float(allZ.max()))
        cl = (float(min(allC.min(), 0.0)), float(allC.max()))
        normA = colors.LogNorm(vmin=max(zl[0], 1e-3), vmax=zl[1])
        normC = colors.TwoSlopeNorm(vmin=-max(abs(cl[0]), 1e-6), vcenter=0.0, vmax=max(cl[1], 1e-6))
        fig = plt.figure(figsize=(13, 9.8))
        for r_, (kind, ZZ, norm, cmap, zlim, zlab) in enumerate((("absolute", Zs, normA, cm.viridis, zl, "CE"),
                                                                 ("centred", Cs, normC, cm.RdBu_r, cl, "CE − centre CE"))):
            for c_, m in enumerate(C.METHODS):
                ax = fig.add_subplot(3, 4, r_ * 4 + c_ + 1, projection="3d")
                panel3d(ax, grid, ZZ[m], norm, cmap, zlim, "%s, seed 0 (%s)" % (LAB[m], kind), zlab if c_ == 0 else "",
                        g2=grid2)
                ax.set_xlabel("t along %s (r)" % v1, fontsize=6.5); ax.set_ylabel("t along %s (r)" % v2, fontsize=6.5)
        i0 = int(np.argmin(np.abs(np.asarray(grid))))
        for c_, m in enumerate(C.METHODS):
            ax = fig.add_subplot(3, 4, 8 + c_ + 1)
            ax.plot(grid, Zs[m][:, i0], color="C3", lw=1, label="along %s (t₂ = 0), T₁ = %.2g" % (v1, ext[0]))
            ax.plot(grid2, Zs[m][i0, :], color="C0", lw=1, label="along %s (t₁ = 0), T₂ = %.2g" % (v2, ext[1]))
            ax.set_xscale("symlog", linthresh=min(ext) / 10)
            ax.set_yscale("log"); ax.set_ylim(*zl); ax.grid(alpha=0.3)
            ax.set_xlabel("t (r)"); ax.set_title("measured cuts through the grid", fontsize=7)
            if c_ == 0:
                ax.set_ylabel("CE"); ax.legend(frameon=False, fontsize=6)
        fig.suptitle("Seed 0, plane of relative-coordinate eigendirections %s/%s (train-probe Hessian, %s), %s. Mapped directions are "
                     "scaled to r = 1 and are not orthogonal in weight space." % (v1, v2, C.POLICY_LABEL[pol], SPL[split]), fontsize=8)
        save(fig, "E_%s_surface3d_%s" % (name, split))
        fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
        for r_, (kind, ZZ, norm, cmap, zlim, zlab) in enumerate((("absolute", Zs, normA, cm.viridis, zl, "CE"),
                                                                 ("centred", Cs, normC, cm.RdBu_r, cl, "CE − centre CE"))):
            for c_, m in enumerate(C.METHODS):
                ax = axes[r_, c_]
                levels = np.geomspace(max(zlim[0], 1e-3), zlim[1], 14) if kind == "absolute" else np.linspace(zlim[0] if zlim[0] < 0 else -1e-6, zlim[1], 15)
                cs = ax.contourf(grid, grid2, np.ma.masked_invalid(ZZ[m]).T, levels=levels, cmap=cmap, norm=norm)
                ax.contour(grid, grid2, np.ma.masked_invalid(ZZ[m]).T, levels=levels, colors="k", linewidths=0.25)
                ax.plot(0, 0, "k+"); ax.set_aspect("auto")
                ax.set_title("%s (%s)" % (LAB[m], kind), fontsize=7.5)
                ax.set_xlabel("t, %s" % v1, fontsize=7); ax.set_ylabel("t, %s" % v2, fontsize=7)
            fig.colorbar(cs, ax=axes[r_, :], shrink=0.8, label=zlab)
        fig.suptitle("Seed 0 Hessian-eigendirection plane %s/%s, %s, %s (contours)" % (v1, v2, C.POLICY_LABEL[pol], SPL[split]))
        save(fig, "E_%s_contour_%s" % (name, split))
    return rows


def interactive(D):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return "plotly unavailable"
    specs = [("random 0/1, recalibrated at every point", C.WIDE41, lambda m, a, b: C.spec_r2(0, 1, a, b), C.POINTWISE)]
    for pol in C.HESS_POLICIES:
        for v1, v2 in (("top1", "top2"), ("top1", "min")):
            specs.append(("Hessian %s/%s, %s (unit grid u; t_i = u T_i, see figure captions)" % (v1, v2, C.POLICY_LABEL[pol]),
                          C.HESS_PLANE_U,
                          lambda m, a, b, pol=pol, v1=v1, v2=v2: C.spec_h2(C.hess_problem(m, 0, "train_probe", pol, "relative"), v1, v2, a, b), pol))
    html = ["<html><head><meta charset='utf-8'><title>landscape v3 surfaces</title></head><body>",
            "<p>Seed 0, test probe, final state. Every vertex is measured. Absolute CE, shared z-range per row.</p>"]
    for title, grid, fn, pol in specs:
        fig = make_subplots(rows=1, cols=4, specs=[[{"type": "surface"}] * 4], subplot_titles=[LAB[m] for m in C.METHODS])
        Zs = {m: grid_values(D, m, 0, lambda a, b, m=m: fn(m, a, b), grid, pol, "test_probe") for m in C.METHODS}
        good = [z[np.isfinite(z)] for z in Zs.values() if np.isfinite(z).any()]
        if not good:
            continue
        lo, hi = float(np.concatenate(good).min()), float(np.concatenate(good).max())
        for c_, m in enumerate(C.METHODS):
            fig.add_trace(go.Surface(x=list(grid), y=list(grid), z=Zs[m].T, cmin=lo, cmax=hi, colorscale="Viridis",
                                     showscale=c_ == 3), row=1, col=c_ + 1)
        scene = dict(zaxis=dict(range=[lo, hi]), xaxis_title="t1", yaxis_title="t2", zaxis_title="CE",
                     camera=dict(eye=dict(x=-1.4, y=-1.4, z=0.9)))
        fig.update_layout(title=title, height=460, **{"scene%s" % ("" if i == 0 else i + 1): scene for i in range(4)})
        html.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if len(html) == 2 else False))
    html.append("</body></html>")
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "surfaces_interactive.html").write_text("\n".join(html), encoding="utf-8")
    return "ok"

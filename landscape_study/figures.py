"""Figures and summary numbers from the raw results.

    py -m landscape_study.figures --results studies/landscape_v1/results

Reads every ``*.jsonl`` under ``--results`` (any sub-folder), writes PDF+PNG to
``studies/landscape_v1/figures/`` and ``summary.json`` next to them.  Missing
tasks and non-finite values are reported in ``summary.json``, never dropped
silently.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "studies" / "landscape_v1" / "figures"
INPUTS = ROOT / "studies" / "landscape_v1" / "inputs"
COL = {"plain": "#555555", "resolution_max_b1": "#1f77b4"}
LAB = {"plain": "plain", "resolution_max_b1": "resolution-only (block-1 max-pool)"}
SPL = {"train_probe": "train probe (1,000)", "test_probe": "test probe (1,000)"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 150})


def load(results: Path) -> dict:
    tasks = defaultdict(list)
    for p in sorted(results.rglob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                tasks[p.stem].append(json.loads(line))
    # a task may appear in several folders (resume); keep the last value per key
    out = {}
    for k, rows in tasks.items():
        by = {}
        for r in rows:
            by[r["key"]] = r
        out[k] = list(by.values())
    return out


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(FIG / (name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def grid_array(rows, split, field="ce"):
    a = sorted({r["a"] for r in rows})
    b = sorted({r["b"] for r in rows})
    Z = np.full((len(b), len(a)), np.nan)
    for r in rows:
        Z[b.index(r["b"]), a.index(r["a"])] = r["splits"][split][field]
    return np.array(a), np.array(b), Z


def center_of(rows, split):
    c = [r for r in rows if r["a"] == 0 and r.get("b", 0) in (0, None)]
    return c[0]["splits"][split]["ce"] if c else np.nan


def nonfinite(tasks):
    bad = {}
    for k, rows in tasks.items():
        n = sum(1 for r in rows for s in r["splits"].values()
                if not (s["finite"] and np.isfinite(s["ce"])))
        if n:
            bad[k] = n
    return bad


def panel_grid(ax, a, b, Z, vmin, vmax, cmap, levels):
    im = ax.imshow(Z, origin="lower", extent=[a[0], a[-1], b[0], b[-1]], vmin=vmin, vmax=vmax,
                   cmap=cmap, aspect="equal", interpolation="nearest")
    A, B = np.meshgrid(a, b)
    cs = ax.contour(A, B, Z, levels=levels, colors="k", linewidths=0.5)
    ax.clabel(cs, fontsize=6, fmt="%.2g")
    ax.plot(0, 0, "w+", ms=6)
    return im


# -- A ----------------------------------------------------------------------

def fig_A(tasks, summary):
    ms = ("plain", "resolution_max_b1")
    keys = {m: "A2d__%s__seed0" % m for m in ms}
    missing = [k for k in keys.values() if k not in tasks]
    summary["A2d_missing"] = missing
    if missing:
        return
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), constrained_layout=True)
    for col, split in enumerate(("train_probe", "test_probe")):
        arrs = {m: grid_array(tasks[keys[m]], split) for m in ms}
        vmax = max(np.nanmax(Z) for _, _, Z in arrs.values())
        vmin = min(np.nanmin(Z) for _, _, Z in arrs.values())
        dmax = max(np.nanmax(Z - center_of(tasks[keys[m]], split)) for m, (_, _, Z) in arrs.items())
        for row, m in enumerate(ms):
            a, b, Z = arrs[m]
            c0 = center_of(tasks[keys[m]], split)
            ax = axes[row, 2 * col]
            im = panel_grid(ax, a, b, Z, vmin, vmax, "viridis", np.linspace(vmin, vmax, 9)[1:-1])
            short = "plain" if m == "plain" else "res.-only"
            ax.set_title("%s, %s\nCE (centre %.3f)" % (short, split.replace("_", " "), c0))
            if row == 1:
                fig.colorbar(im, ax=axes[:, 2 * col], shrink=0.6, label="CE")
            ax = axes[row, 2 * col + 1]
            im2 = panel_grid(ax, a, b, Z - c0, 0, dmax, "magma", [0.05, 0.1, 0.25, 0.5, 1.0])
            ax.set_title("%s, %s\nCE - centre" % (short, split.replace("_", " ")))
            if row == 1:
                fig.colorbar(im2, ax=axes[:, 2 * col + 1], shrink=0.6, label="increase in CE")
            summary.setdefault("A2d", {}).setdefault(m, {})[split] = {
                "center_ce": c0, "max_ce": float(np.nanmax(Z)),
                "mean_increase_over_grid": float(np.nanmean(Z - c0)),
                "increase_at_corners_mean": float(np.mean([Z[0, 0], Z[0, -1], Z[-1, 0], Z[-1, -1]]) - c0),
                "center_acc": [r for r in tasks[keys[m]] if r["a"] == 0 and r["b"] == 0][0]["splits"][split]["acc"]}
    for ax in axes.flat:
        ax.set_xlabel("a (direction D)")
        ax.set_ylabel("b (direction E)")
    fig.suptitle("Analysis A. Epoch-30 solutions, seed 0, direction pair 0 (matched relative "
                 "perturbations, not one physical plane). Target state, recalibrated BatchNorm.")
    save(fig, "A_local_landscapes_seed0")


def slice_rows(rows, split):
    rows = sorted(rows, key=lambda r: r["a"])
    a = np.array([r["a"] for r in rows])
    ce = np.array([r["splits"][split]["ce"] for r in rows])
    return a, ce - ce[np.argmin(np.abs(a))], ce


def fig_A1d(tasks, summary):
    ms = ("plain", "resolution_max_b1")
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4), sharex=True, sharey="row", constrained_layout=True)
    table = {}
    for col, seed in enumerate((0, 1, 2)):
        for row, split in enumerate(("train_probe", "test_probe")):
            ax = axes[row, col]
            for m in ms:
                ids = ["A1d__%s__seed%d__pair%d__D" % (m, seed, p) for p in range(5)] + \
                      ["A1d__%s__seed%d__pair0__E" % (m, seed)]
                for tid in ids:
                    if tid not in tasks:
                        table.setdefault("missing", []).append(tid)
                        continue
                    a, d, ce = slice_rows(tasks[tid], split)
                    ax.plot(a, d, color=COL[m], lw=0.9, alpha=0.8,
                            label=LAB[m] if tid == ids[0] else None)
                    i1 = [np.argmin(np.abs(a - v)) for v in (-0.1, 0.1)]
                    i2 = [np.argmin(np.abs(a - v)) for v in (-0.25, 0.25)]
                    table.setdefault(m, {}).setdefault("seed%d" % seed, {}).setdefault(split, []).append(
                        {"slice": tid.split("__", 3)[-1], "center_ce": float(ce[np.argmin(np.abs(a))]),
                         "increase_at_0.1": float(d[i1].mean()), "increase_at_0.25": float(d[i2].mean())})
            ax.set_title("seed %d, %s" % (seed, SPL[split]))
            if row == 1:
                ax.set_xlabel("a (relative filter-wise step)")
            if col == 0:
                ax.set_ylabel("CE - centre")
            ax.axhline(0, color="k", lw=0.4)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Robustness: 1-D slices along 6 filter-normalised directions per solution "
                 "(pairs 0-4 axis D, pair 0 axis E), epoch 30, target state, recalibrated BN")
    save(fig, "A_robustness_1d_slices")
    # paired summary: per seed, mean over slices of increase at 0.25, method difference
    paired = {}
    for split in ("train_probe", "test_probe"):
        for seed in (0, 1, 2):
            try:
                p = [s["increase_at_0.25"] for s in table["plain"]["seed%d" % seed][split]]
                r = [s["increase_at_0.25"] for s in table["resolution_max_b1"]["seed%d" % seed][split]]
            except KeyError:
                continue
            diffs = [x - y for x, y in zip(p, r)]
            paired.setdefault(split, {})["seed%d" % seed] = {
                "plain_mean_increase_0.25": float(np.mean(p)),
                "resolution_mean_increase_0.25": float(np.mean(r)),
                "plain_minus_resolution_per_slice": diffs,
                "n_slices_plain_higher": int(sum(d > 0 for d in diffs)), "n_slices": len(diffs)}
    table["paired_increase_0.25"] = paired
    # consistency between the 2-D grid and its own 1-D slice (same draws)
    cons = {}
    for m in ms:
        g, l = "A2d__%s__seed0" % m, "A1d__%s__seed0__pair0__D" % m
        if g in tasks and l in tasks:
            gr = {r["a"]: r["splits"]["test_probe"]["ce"] for r in tasks[g] if r["b"] == 0.0}
            lr = {r["a"]: r["splits"]["test_probe"]["ce"] for r in tasks[l]}
            cons[m] = max(abs(gr[k] - lr[k]) for k in lr if k in gr)
    table["max_abs_diff_2d_row_b0_vs_1d_slice_test"] = cons
    summary["A1d"] = table


# -- B ----------------------------------------------------------------------

def fig_B(tasks, summary):
    ss = ("r16", "r24", "r32")
    keys = ["B2d__resolution_max_b1__seed0__ep06__%s" % s for s in ss]
    miss = [k for k in keys if k not in tasks]
    summary["B2d_missing"] = miss
    if miss:
        return
    fig, axes = plt.subplots(4, 3, figsize=(8.4, 11), constrained_layout=True)
    out = {}
    for r0, split in enumerate(("train_probe", "test_probe")):
        arrs = [grid_array(tasks[k], split) for k in keys]
        vmin = min(np.nanmin(Z) for *_, Z in arrs)
        vmax = max(np.nanmax(Z) for *_, Z in arrs)
        cs = [center_of(tasks[k], split) for k in keys]
        dmax = max(np.nanmax(np.abs(Z - c)) for (_, _, Z), c in zip(arrs, cs))
        for c, ((a, b, Z), s, c0) in enumerate(zip(arrs, ss, cs)):
            ax = axes[2 * r0, c]
            im = panel_grid(ax, a, b, Z, vmin, vmax, "viridis", np.linspace(vmin, vmax, 9)[1:-1])
            ax.set_title("%s, evaluated at %s\nCE (centre %.3f)" % (SPL[split], s.replace("r", "r="), c0))
            ax = axes[2 * r0 + 1, c]
            # diverging: points below the centre stay visible (the centre need not be a minimum)
            im2 = panel_grid(ax, a, b, Z - c0, -dmax, dmax, "RdBu_r",
                             [-0.25, -0.1, 0.1, 0.25, 0.5])
            ax.set_title("%s, %s: CE - centre" % (SPL[split], s.replace("r", "r=")))
            out.setdefault(s, {})[split] = {"center_ce": c0, "mean_increase_over_grid": float(np.nanmean(Z - c0)),
                                            "max_increase": float(np.nanmax(Z - c0))}
        fig.colorbar(im, ax=axes[2 * r0, :], shrink=0.7, label="CE")
        fig.colorbar(im2, ax=axes[2 * r0 + 1, :], shrink=0.7, label="CE - centre (blue: below centre)")
    for ax in axes.flat:
        ax.set_xlabel("a (D)")
        ax.set_ylabel("b (E)")
    fig.suptitle("Analysis B. Same weights (resolution-only, seed 0, epoch 6: last update at r=16),\n"
                 "same centre, directions, subsets and axes; three counterfactual reduction states. "
                 "Recalibrated BN per state.")
    save(fig, "B_fixed_weights_resolution_states")
    summary["B2d"] = out

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    b1 = {}
    for ax, split in zip(axes, ("train_probe", "test_probe")):
        for who, ls in (("resolution_max_b1__seed0__ep12", "-"), ("plainweights_with_hook__seed0__ep06", ":")):
            for s, colr in zip(ss, ("#d62728", "#ff7f0e", "#2ca02c")):
                k = "B1d__%s__%s" % (who, s)
                if k not in tasks:
                    b1.setdefault("missing", []).append(k)
                    continue
                a, d, ce = slice_rows(tasks[k], split)
                ax.plot(a, ce, ls=ls, color=colr, label="%s, %s" % (
                    "res. ep12" if "ep12" in who else "plain ep6 + hook", s.replace("r", "r=")))
                b1.setdefault(k, {})[split] = {"center_ce": float(ce[np.argmin(np.abs(a))]),
                                               "increase_at_0.25": float(d[[0, -1]].mean())}
        ax.set_title(SPL[split])
        ax.set_xlabel("a (direction D, pair 0)")
        ax.set_ylabel("CE")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Analysis B, secondary 1-D slices: resolution-only epoch 12 (last update at r=24), "
                 "and plain epoch-6 weights evaluated with the reduction hook (control)")
    save(fig, "B_secondary_1d")
    summary["B1d"] = b1


# -- C ----------------------------------------------------------------------

def fig_C(tasks, summary):
    pl = torch.load(INPUTS / "pca_plane.pt", map_location="cpu", weights_only=False)
    if "C_plane" not in tasks:
        summary["C_missing"] = ["C_plane"]
        return
    xs, ys = np.array(pl["grid"]["x"]), np.array(pl["grid"]["y"])
    ck = {(r["file"], r["bn_policy"], r["state_label"]): r for r in tasks.get("C_checkpoints", [])}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    co = np.array(pl["coords"])
    for ax, split in zip(axes, ("train_probe", "test_probe")):
        Z = np.full((len(ys), len(xs)), np.nan)
        for r in tasks["C_plane"]:
            Z[r["j"], r["i"]] = r["splits"][split]["ce"]
        Zc = np.minimum(Z, np.nanpercentile(Z, 95))
        X, Y = np.meshgrid(xs, ys)
        cf = ax.contourf(X, Y, np.log10(Zc), levels=20, cmap="viridis")
        fig.colorbar(cf, ax=ax, label="log10 CE (clipped at 95th pct)")
        for m in ("plain", "resolution_max_b1"):
            idx = [i for i, p in enumerate(pl["points"]) if p["method"] == m]
            ax.plot(co[idx, 0], co[idx, 1], "-o", color="w" if m == "plain" else "#ff7f0e",
                    mec="k", ms=5, lw=1.5, label=LAB[m])
            for i in idx:
                p = pl["points"][i]
                ax.annotate("%d" % p["epochs_completed"], co[i], textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color="w")
        ax.set_title("%s: target objective on the PCA plane" % SPL[split])
        ax.set_xlabel("PC1 (%.1f%% of variance)" % (100 * pl["explained_variance_ratio"][0]))
        ax.set_ylabel("PC2 (%.1f%%)" % (100 * pl["explained_variance_ratio"][1]))
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Analysis C. Seed-0 trajectories (labels = epochs completed; resolution-only used r=16 "
                 "for epochs 0-5, r=24 for 6-11, 32 after). Background = loss of the reconstructed "
                 "in-plane point, not of the off-plane checkpoints.")
    save(fig, "C_pca_trajectories")
    rows = []
    for i, p in enumerate(pl["points"]):
        f = p["file"]
        rt = ck.get((f, "recalibrated", "target"))
        su = ck.get((f, "saved_stats", "used")) or ck.get((f, "saved_stats", "target"))
        rows.append({"file": f, "method": p["method"], "epochs_completed": p["epochs_completed"],
                     "global_update": p["global_update"], "lr_last_update": p["lr_last_update"],
                     "state_used": p["states"]["used_for_last_update"],
                     "pc1": co[i, 0], "pc2": co[i, 1], "residual_norm": pl["residual_norm"][i],
                     "relative_residual": pl["relative_residual"][i],
                     "checkpoint_ce_recalibrated_target": None if rt is None else
                     {s: rt["splits"][s]["ce"] for s in ("train_probe", "test_probe")},
                     "checkpoint_ce_saved_stats_own_state": None if su is None else
                     {s: su["splits"][s]["ce"] for s in ("train_probe", "test_probe")}})
    summary["C"] = {"explained_variance_ratio": pl["explained_variance_ratio"],
                    "plane_variance_ratio": pl["plane_variance_ratio"], "points": rows,
                    "lr_transitions": "linear warm-up over updates 0-59, then cosine decay to 0 at "
                                      "update 11730; no step change. All checkpoints after warm-up."}


# -- D ----------------------------------------------------------------------

def fig_D(tasks, summary):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    out = {}
    for ax, split in zip(axes, ("train_probe", "test_probe")):
        for seed, ls in zip((0, 1, 2), ("-", "--", ":")):
            k = "D_interp__plain_vs_resolution_max_b1__seed%d" % seed
            if k not in tasks:
                out.setdefault("missing", []).append(k)
                continue
            rows = sorted(tasks[k], key=lambda r: r["alpha"])
            al = np.array([r["alpha"] for r in rows])
            ce = np.array([r["splits"][split]["ce"] for r in rows])
            acc = np.array([r["splits"][split]["acc"] for r in rows])
            ax.plot(al, ce, ls=ls, color="k", label="seed %d" % seed)
            out.setdefault("seed%d" % seed, {})[split] = {
                "L_plain_alpha0": float(ce[0]), "L_resolution_alpha1": float(ce[-1]),
                "max_L": float(ce.max()), "argmax_alpha": float(al[ce.argmax()]),
                "barrier_grid_estimate": float(ce.max() - max(ce[0], ce[-1])),
                "min_acc": float(acc.min()), "acc_alpha0": float(acc[0]), "acc_alpha1": float(acc[-1]),
                "n_points": int(len(al))}
        ax.set_title(SPL[split])
        ax.set_xlabel("alpha (0 = plain, 1 = resolution-only)")
        ax.set_ylabel("CE")
        ax.set_yscale("log")
    axes[0].legend(frameon=False)
    fig.suptitle("Analysis D. Straight segments between paired epoch-30 solutions, target state, "
                 "recalibrated BN, 51 points")
    save(fig, "D_interpolation")
    summary["D"] = out


def saved_vs_recal(tasks, summary):
    rows = tasks.get("S_saved_vs_recalibrated", [])
    by = defaultdict(dict)
    for r in rows:
        by[r["file"]][(r["bn_policy"], r["state_label"])] = r
    out = {}
    for f, d in sorted(by.items()):
        rec = next(iter(d.values()))["recorded_metrics"]
        e = {}
        for (pol, lab), r in d.items():
            e["%s|%s" % (pol, lab)] = {s: {"ce": r["splits"][s]["ce"], "acc": r["splits"][s]["acc"]}
                                       for s in r["splits"]}
        if rec:
            e["recorded"] = {"current": {"test_ce": rec["test_ce_current"], "test_acc": rec["test_acc_current"],
                                         "train_probe_ce": rec["train_probe_ce_current"]},
                             "bypass32": {"test_ce": rec["test_ce_bypass32"], "test_acc": rec["test_acc_bypass32"],
                                          "train_probe_ce": rec["train_probe_ce_bypass32"]}}
            own = d.get(("saved_stats", "used")) or d.get(("saved_stats", "target"))
            e["reproduction_abs_error_current"] = {
                "test_ce": abs(own["splits"]["test_full"]["ce"] - rec["test_ce_current"]),
                "test_acc": abs(own["splits"]["test_full"]["acc"] - rec["test_acc_current"]),
                "train_probe_ce": abs(own["splits"]["pinned_train_probe_500"]["ce"] - rec["train_probe_ce_current"])}
        out[f] = e
    summary["saved_vs_recalibrated"] = out
    if out:
        errs = [v["reproduction_abs_error_current"] for v in out.values() if "reproduction_abs_error_current" in v]
        summary["reproduction_max_abs_error"] = {k: max(e[k] for e in errs) for k in errs[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT / "studies" / "landscape_v1" / "results"))
    a = ap.parse_args()
    tasks = load(Path(a.results))
    summary = {"tasks_found": sorted(tasks), "non_finite_counts": nonfinite(tasks)}
    for f in (fig_A, fig_A1d, fig_B, fig_C, fig_D, saved_vs_recal):
        f(tasks, summary)
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("figures and summary in", FIG)


if __name__ == "__main__":
    main()

"""Aggregate the v2 raw results into tables, figures and an inventory.

    py -m landscape_v2.analyze [--raw studies/landscape_v2/raw]

Reads ``raw/<account>/v2/`` for every account, writes ``studies/landscape_v2/
figures/`` (PDF + PNG + ``surfaces_interactive.html``) and ``tables/`` (CSV +
``summary.json``).  Every aggregate is computed per training seed first; seeds
are the replication unit.  Missing tasks, deadline stops, failures and
non-finite values are listed in ``tables/summary.json`` and never dropped
silently.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from . import common as C

FIG = C.STUDY / "figures"
TAB = C.STUDY / "tables"
COL = {"plain": "#444444", "resolution_max_b1": "#1f77b4", "gaussian_postrelu": "#ff7f0e",
       "resolution_max_b1_gaussian_conv": "#9467bd"}
LAB = {"plain": "plain", "resolution_max_b1": "resolution-only", "gaussian_postrelu": "Gaussian-only",
       "resolution_max_b1_gaussian_conv": "combined"}
SPL = {"train_probe": "train probe (1,000)", "test_probe": "test probe (1,000)",
       "train_large": "train subset (10,000)", "test_full": "full test set (10,000)",
       "train_full": "full train set (50,000)"}
POL = {"saved": "saved BN statistics", "recalibrated": "recalibrated BN statistics"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "figure.dpi": 150})
ELEV, AZIM, BOX = 28, -58, (1, 1, 0.75)


# -- loading ------------------------------------------------------------------

class Raw:
    def __init__(self, root: Path):
        self.root = root
        self.v2 = sorted(p for p in root.glob("*/v2") if p.is_dir())
        self.issues = []

    def task(self, tid):
        rows, metas = {}, []
        for d in self.v2:
            p = d / "eval" / (tid + ".jsonl")
            if p.exists():
                for line in p.read_text().splitlines():
                    try:
                        r = json.loads(line)
                        rows[r["key"]] = r
                    except Exception:
                        self.issues.append({"task": tid, "problem": "unparsable line (interrupted write?)"})
            mp = d / "eval" / (tid + ".meta.json")
            if mp.exists():
                metas.append(json.loads(mp.read_text()))
        return rows, metas

    def file(self, rel):
        for d in self.v2:
            if (d / rel).exists():
                return d / rel
        return None


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(FIG / (name + ".png"), bbox_inches="tight", dpi=170)
    plt.close(fig)


def write_csv(name, rows):
    TAB.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0])
    with open(TAB / (name + ".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def sd(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


# -- 1. primary sensitivity ------------------------------------------------------

def sensitivity(raw, summary):
    per_dir, seed_rows, status = [], [], {}
    S = {}      # (m, s, pol, split, amp) -> list over directions of S_k
    for s in C.SEEDS:
        for m in C.METHODS:
            tid = "sens1d__%s__seed%d" % (C.SHORT[m], s)
            rows, metas = raw.task(tid)
            status[tid] = metas[-1]["status"] if metas else "missing"
            for pol in ("saved", "recalibrated"):
                c = rows.get("%s|center" % pol)
                if c is None:
                    continue
                for k in range(C.N_DIRECTIONS):
                    for amp in C.AMPLITUDES:
                        p, q = rows.get("%s|%d|%g|1" % (pol, k, amp)), rows.get("%s|%d|%g|-1" % (pol, k, amp))
                        if p is None or q is None:
                            continue
                        for sp in ("train_probe", "test_probe"):
                            l0, lp, lq = c["splits"][sp]["ce"], p["splits"][sp]["ce"], q["splits"][sp]["ce"]
                            val = 0.5 * (lp + lq) - l0
                            S.setdefault((m, s, pol, sp, amp), []).append(val)
                            per_dir.append({"method": m, "seed": s, "policy": pol, "split": sp, "direction": k,
                                            "direction_seed": C.direction_seed(s, k), "amplitude": amp,
                                            "L_center": l0, "dL_plus": lp - l0, "dL_minus": lq - l0, "S": val,
                                            "acc_center": c["splits"][sp]["acc"],
                                            "acc_plus": p["splits"][sp]["acc"], "acc_minus": q["splits"][sp]["acc"]})
    write_csv("sensitivity_per_direction", per_dir)
    for (m, s, pol, sp, amp), v in sorted(S.items()):
        seed_rows.append({"method": m, "seed": s, "policy": pol, "split": sp, "amplitude": amp,
                          "n_directions": len(v), "S_mean": float(np.mean(v)), "S_direction_sd": sd(v),
                          "S_min": float(np.min(v)), "S_max": float(np.max(v))})
    write_csv("sensitivity_per_seed", seed_rows)
    paired = []
    for pol in ("saved", "recalibrated"):
        for sp in ("train_probe", "test_probe"):
            for amp in C.AMPLITUDES:
                for m in C.METHODS[1:]:
                    diffs, per_seed = [], {}
                    for s in C.SEEDS:
                        a, b = S.get((m, s, pol, sp, amp)), S.get(("plain", s, pol, sp, amp))
                        if a and b and len(a) == len(b):
                            d = float(np.mean(a) - np.mean(b))
                            diffs.append(d)
                            per_seed[s] = d
                    paired.append({"policy": pol, "split": sp, "amplitude": amp, "method": m,
                                   "n_seeds": len(diffs), "mean_diff_vs_plain": float(np.mean(diffs)) if diffs else float("nan"),
                                   "sd_across_seeds": sd(diffs), "n_seeds_lower_than_plain": int(sum(d < 0 for d in diffs)),
                                   **{"seed%d" % s: per_seed.get(s, float("nan")) for s in C.SEEDS}})
    write_csv("sensitivity_paired_vs_plain", paired)
    summary["sensitivity_task_status"] = status
    summary["sensitivity_paired"] = paired

    # figure: seed means by amplitude
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, constrained_layout=True)
    for i, pol in enumerate(("saved", "recalibrated")):
        for j, sp in enumerate(("train_probe", "test_probe")):
            ax = axes[i, j]
            for m in C.METHODS:
                means = []
                for amp in C.AMPLITUDES:
                    vals = [np.mean(S[(m, s, pol, sp, amp)]) for s in C.SEEDS if (m, s, pol, sp, amp) in S]
                    ax.scatter([amp] * len(vals), vals, color=COL[m], s=6, alpha=0.35)
                    means.append((np.mean(vals) if vals else np.nan, sd(vals)))
                mu = np.array([x[0] for x in means]); sg = np.array([x[1] for x in means])
                ax.errorbar(C.AMPLITUDES, mu, yerr=sg, color=COL[m], marker="o", ms=3, lw=1.2, capsize=2, label=LAB[m])
            ax.set_yscale("log")
            ax.set_title("%s, %s" % (POL[pol], SPL[sp]))
            ax.set_xlabel("relative amplitude ε")
            ax.set_ylabel("S̄(ε): mean over 20 directions")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Symmetric finite-perturbation sensitivity of the final solutions (final state). Dots: the five "
                 "training seeds (each averaged over 20 directions);\nline: mean over seeds ± SD across seeds. "
                 "Log scale.")
    save(fig, "S1_sensitivity_by_amplitude")

    # figure: paired differences per seed
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, constrained_layout=True)
    for i, pol in enumerate(("saved", "recalibrated")):
        for j, sp in enumerate(("train_probe", "test_probe")):
            ax = axes[i, j]
            for mi, m in enumerate(C.METHODS[1:]):
                for amp_i, amp in enumerate(C.AMPLITUDES):
                    row = [r for r in paired if r["policy"] == pol and r["split"] == sp and r["amplitude"] == amp and r["method"] == m][0]
                    x = amp_i + (mi - 1) * 0.22
                    vals = [row["seed%d" % s] for s in C.SEEDS]
                    ax.scatter([x] * 5, vals, color=COL[m], s=10, alpha=0.7, label=LAB[m] if amp_i == 0 else None)
                    ax.plot([x - 0.08, x + 0.08], [row["mean_diff_vs_plain"]] * 2, color=COL[m], lw=2)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(range(len(C.AMPLITUDES)), [str(a) for a in C.AMPLITUDES])
            ax.set_title("%s, %s" % (POL[pol], SPL[sp]))
            ax.set_xlabel("relative amplitude ε")
            ax.set_ylabel("S̄(method) − S̄(plain), same seed")
            ax.grid(alpha=0.3, axis="y")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Paired differences against plain, computed within each training seed (dots = seeds, bar = mean). "
                 "Below 0: smaller loss increase than plain.")
    save(fig, "S2_paired_differences")

    # figure: directional spread at 0.10 and 0.25, test probe, both policies
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey="row", constrained_layout=True)
    for i, amp in enumerate((0.10, 0.25)):
        for j, pol in enumerate(("saved", "recalibrated")):
            ax = axes[i, j]
            for mi, m in enumerate(C.METHODS):
                for s in C.SEEDS:
                    v = S.get((m, s, pol, "test_probe", amp))
                    if not v:
                        continue
                    x = mi * 6 + s
                    ax.boxplot(v, positions=[x], widths=0.7, showfliers=True, patch_artist=True,
                               boxprops=dict(facecolor=COL[m], alpha=0.4), medianprops=dict(color="k"),
                               flierprops=dict(markersize=2))
            ax.set_xticks([mi * 6 + 2 for mi in range(4)], [LAB[m] for m in C.METHODS])
            ax.set_yscale("log")
            ax.set_title("test probe, ε = %g, %s (boxes: 20 directions per seed 0-4)" % (amp, POL[pol]))
            ax.set_ylabel("S for one direction")
    fig.suptitle("Direction-to-direction spread of the symmetric sensitivity (within each training seed)")
    save(fig, "S3_direction_spread")
    return S


# -- 2. validation and calibration sensitivity --------------------------------------

def validation(raw, S, summary):
    rows_c, rows_p = [], []
    for s in C.SEEDS:
        for m in C.METHODS:
            v, _ = raw.task("validation__%s__seed%d" % (C.SHORT[m], s))
            probe, _ = raw.task("sens1d__%s__seed%d" % (C.SHORT[m], s))
            for pol in ("saved", "recalibrated"):
                c = v.get("%s|center" % pol)
                if c:
                    rows_c.append({"method": m, "seed": s, "policy": pol,
                                   **{"%s_%s" % (sp, q): c["splits"][sp][q] for sp in ("train_full", "test_full", "train_large")
                                      for q in ("ce", "acc")}})
                pc = probe.get("%s|center" % pol)
                for k in C.VALIDATION["directions"]:
                    for amp in C.VALIDATION["amplitudes"]:
                        p, q = v.get("%s|%d|%g|1" % (pol, k, amp)), v.get("%s|%d|%g|-1" % (pol, k, amp))
                        pp, pq = probe.get("%s|%d|%g|1" % (pol, k, amp)), probe.get("%s|%d|%g|-1" % (pol, k, amp))
                        if not (c and p and q):
                            continue
                        rec = {"method": m, "seed": s, "policy": pol, "direction": k, "amplitude": amp}
                        for sp, psp in (("train_large", "train_probe"), ("test_full", "test_probe")):
                            rec["S_" + sp] = 0.5 * (p["splits"][sp]["ce"] + q["splits"][sp]["ce"]) - c["splits"][sp]["ce"]
                            rec["S_" + psp] = (0.5 * (pp["splits"][psp]["ce"] + pq["splits"][psp]["ce"]) - pc["splits"][psp]["ce"]
                                               if pp and pq and pc else float("nan"))
                        rows_p.append(rec)
    write_csv("validation_centers_full", rows_c)
    write_csv("validation_points", rows_p)
    agree = []
    for pol in ("saved", "recalibrated"):
        for amp in C.VALIDATION["amplitudes"]:
            for m in C.METHODS[1:]:
                for big, small in (("train_large", "train_probe"), ("test_full", "test_probe")):
                    db, ds = [], []
                    for s in C.SEEDS:
                        a = [r for r in rows_p if r["method"] == m and r["seed"] == s and r["policy"] == pol and r["amplitude"] == amp]
                        b = [r for r in rows_p if r["method"] == "plain" and r["seed"] == s and r["policy"] == pol and r["amplitude"] == amp]
                        if len(a) == len(b) == 2:
                            db.append(np.mean([r["S_" + big] for r in a]) - np.mean([r["S_" + big] for r in b]))
                            ds.append(np.mean([r["S_" + small] for r in a]) - np.mean([r["S_" + small] for r in b]))
                    agree.append({"policy": pol, "amplitude": amp, "method": m, "set": big, "n_seeds": len(db),
                                  "mean_diff_large": float(np.mean(db)) if db else float("nan"),
                                  "mean_diff_probe_same_points": float(np.mean(ds)) if ds else float("nan"),
                                  "n_seeds_same_sign": int(sum(np.sign(x) == np.sign(y) for x, y in zip(db, ds))),
                                  "n_seeds_lower_than_plain_large": int(sum(x < 0 for x in db))})
    write_csv("validation_probe_vs_large", agree)
    summary["validation_probe_vs_large"] = agree

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, (big, small) in zip(axes, (("train_large", "train_probe"), ("test_full", "test_probe"))):
        for m in C.METHODS:
            for pol, mk in (("saved", "o"), ("recalibrated", "^")):
                r = [x for x in rows_p if x["method"] == m and x["policy"] == pol]
                ax.scatter([x["S_" + small] for x in r], [x["S_" + big] for x in r], s=12, marker=mk,
                           color=COL[m], alpha=0.6, label="%s, %s" % (LAB[m], pol))
        lim = ax.get_xlim()
        ax.plot(lim, lim, "k:", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("S on the 1,000-image %s" % small.replace("_", " "))
        ax.set_ylabel("S on the %s" % SPL[big])
    axes[1].legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Validation of prespecified points (directions 0-1, ε = 0.10 and 0.25, both signs, all methods "
                 "and seeds): probe vs larger evaluation sets")
    save(fig, "V1_probe_vs_larger_sets")

    cal = []
    for s in C.CALIB_SENSITIVITY["seeds"]:
        for m in C.METHODS:
            r10, _ = raw.task("calibsens__%s__seed%d" % (C.SHORT[m], s))
            r2, _ = raw.task("sens1d__%s__seed%d" % (C.SHORT[m], s))
            c10, c2 = r10.get("center"), r2.get("recalibrated|center")
            for k in C.CALIB_SENSITIVITY["directions"]:
                for amp in C.CALIB_SENSITIVITY["amplitudes"]:
                    a10 = [r10.get("%d|%g|%d" % (k, amp, sg)) for sg in (1, -1)]
                    a2 = [r2.get("recalibrated|%d|%g|%d" % (k, amp, sg)) for sg in (1, -1)]
                    if c10 and c2 and all(a10) and all(a2):
                        for sp in ("train_probe", "test_probe"):
                            cal.append({"method": m, "seed": s, "direction": k, "amplitude": amp, "split": sp,
                                        "L0_calib2k": c2["splits"][sp]["ce"], "L0_calib10k": c10["splits"][sp]["ce"],
                                        "S_calib2k": 0.5 * sum(x["splits"][sp]["ce"] for x in a2) - c2["splits"][sp]["ce"],
                                        "S_calib10k": 0.5 * sum(x["splits"][sp]["ce"] for x in a10) - c10["splits"][sp]["ce"]})
    write_csv("calibration_size_sensitivity", cal)
    summary["calibration_size_sensitivity"] = cal


# -- 3. interpolation -------------------------------------------------------------

SEGMENTS = (("plain", "resolution_max_b1"), ("plain", "gaussian_postrelu"),
            ("plain", "resolution_max_b1_gaussian_conv"), ("resolution_max_b1", "resolution_max_b1_gaussian_conv"))


def interpolation(raw, summary):
    rows = []
    fig, axes = plt.subplots(4, 2, figsize=(9, 11), constrained_layout=True)
    for i, (a, b) in enumerate(SEGMENTS):
        for j, sp in enumerate(("train_probe", "test_probe")):
            ax = axes[i, j]
            for s in C.SEEDS:
                r, metas = raw.task("interp__%s_%s__seed%d" % (C.SHORT[a], C.SHORT[b], s))
                if len(r) < 51:
                    if j == 0:
                        rows.append({"A": a, "B": b, "seed": s, "status": "incomplete (%d/51)" % len(r)})
                    continue
                pts = sorted(r.values(), key=lambda x: x["alpha"])
                al = np.array([p["alpha"] for p in pts]); ce = np.array([p["splits"][sp]["ce"] for p in pts])
                acc = np.array([p["splits"][sp]["acc"] for p in pts])
                ax.plot(al, ce, lw=1, label="seed %d" % s)
                if j == 0:
                    rows.append({"A": a, "B": b, "seed": s, "status": "complete"})
                rows[-1] if j == 0 else None
                rec = [x for x in rows if x["A"] == a and x["B"] == b and x["seed"] == s][0]
                rec.update({"%s_L_A" % sp: ce[0], "%s_L_B" % sp: ce[-1], "%s_max" % sp: ce.max(),
                            "%s_argmax_alpha" % sp: al[ce.argmax()],
                            "%s_barrier" % sp: ce.max() - max(ce[0], ce[-1]), "%s_min_acc" % sp: acc.min()})
            ax.set_title("%s (α=0) → %s (α=1), %s" % (LAB[a], LAB[b], SPL[sp]))
            ax.set_xlabel("α"); ax.set_ylabel("CE"); ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Straight segments between final solutions of the same seed (51 points, final state, "
                 "recalibrated BN).\nA barrier on a straight segment does not establish disconnected basins.")
    save(fig, "D1_interpolation_segments")
    write_csv("interpolation_barriers", [{k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                          for k, v in r.items()} for r in rows])
    summary["interpolation"] = rows


# -- 4. fixed weights ---------------------------------------------------------------

def fixed_weight(raw, summary):
    rows = []
    for m in C.METHODS[1:]:
        for tr in C.fixed_weight_transitions(m):
            states = C.fixed_weight_states(m, tr)
            for s in C.SEEDS:
                r, metas = raw.task("fixed1d__%s__seed%d__u%06d" % (C.SHORT[m], s, tr["update"]))
                for lab, st in states:
                    for sp in ("train_probe", "test_probe"):
                        rec = {"method": m, "transition_update": tr["update"], "seed": s, "state_label": lab,
                               "state": json.dumps(st), "split": sp}
                        for pol in ("saved", "recalibrated"):
                            c = r.get("%s|%s|center" % (lab, pol))
                            rec["L_center_" + pol] = c["splits"][sp]["ce"] if c else float("nan")
                        c = r.get("%s|recalibrated|center" % lab)
                        for amp in C.AMPLITUDES:
                            vals = []
                            for k in range(C.FIXED_WEIGHT_DIRECTIONS):
                                p, q = r.get("%s|recalibrated|%d|%g|1" % (lab, k, amp)), r.get("%s|recalibrated|%d|%g|-1" % (lab, k, amp))
                                if c and p and q:
                                    vals.append(0.5 * (p["splits"][sp]["ce"] + q["splits"][sp]["ce"]) - c["splits"][sp]["ce"])
                            rec["S_recal_eps%g" % amp] = float(np.mean(vals)) if vals else float("nan")
                            rec["n_dirs_eps%g" % amp] = len(vals)
                        rows.append(rec)
    write_csv("fixed_weight_states", rows)
    summary["fixed_weight"] = rows
    combos = [(m, tr) for m in C.METHODS[1:] for tr in C.fixed_weight_transitions(m)]
    fig, axes = plt.subplots(2, len(combos), figsize=(3.1 * len(combos), 6.2), constrained_layout=True)
    for j, (m, tr) in enumerate(combos):
        states = [lab for lab, _ in C.fixed_weight_states(m, tr)]
        for i, (field, ylab) in enumerate((("L_center_recalibrated", "centre CE (recalibrated)"),
                                           ("S_recal_eps0.25", "S̄(0.25), recalibrated"))):
            ax = axes[i, j]
            for x, lab in enumerate(states):
                vals = [r[field] for r in rows if r["method"] == m and r["transition_update"] == tr["update"]
                        and r["state_label"] == lab and r["split"] == "test_probe"]
                ax.scatter([x] * len(vals), vals, color=COL[m], s=12)
                if vals:
                    ax.plot([x - 0.2, x + 0.2], [np.nanmean(vals)] * 2, color="k", lw=1.5)
            if i == 0:
                def short(sd_):
                    parts = []
                    if sd_.get("resolution") is not None:
                        parts.append("r=%d" % sd_["resolution"])
                    if sd_.get("sigma") is not None:
                        parts.append("G=%g" % sd_["sigma"])
                    return ", ".join(parts)
                ax.set_xticks(range(len(states)), ["%s\n%s" % (lab, short(sd_)) for lab, sd_ in C.fixed_weight_states(m, tr)], fontsize=6.5)
            else:
                ax.set_xticks(range(len(states)), states)
            ax.set_title("%s, checkpoint at update %d" % (LAB[m], tr["update"]), fontsize=7.5)
            ax.set_ylabel(ylab); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Fixed weights, different intervention states (test probe; dots = seeds, bar = mean). Top: loss at "
                 "the centre. Bottom: symmetric sensitivity around it (10 directions).")
    save(fig, "F1_fixed_weight_states")


# -- 5. trajectories ----------------------------------------------------------------

def trajectories(raw, summary):
    out = {}
    for s in C.SEEDS:
        pz = raw.file("pca/pca_seed%d.npz" % s)
        if pz is None:
            out[s] = "missing PCA"
            continue
        z = np.load(pz, allow_pickle=True)
        co, meth, upd, fit = z["coords"], z["method"], z["update"], z["fit"]
        evr = z["explained_variance_ratio"]
        frac2 = (z["resid2"] ** 2) / np.maximum(z["dist_to_mean"] ** 2, 1e-30)
        out[s] = {"evr_top5": evr[:5].tolist(), "pc12": float(evr[:2].sum()),
                  "max_unexplained_fraction_pc12": float(frac2.max()),
                  "median_unexplained_fraction_pc12": float(np.median(frac2)),
                  "min_dist_to_mean": float(z["dist_to_mean"].min())}
        fig, ax = plt.subplots(figsize=(7.5, 6))
        for m in C.METHODS:
            idx = np.where(meth == m)[0]
            idx = idx[np.argsort(upd[idx])]
            ep = idx[fit[idx]]
            ax.plot(co[idx, 0], co[idx, 1], "-", color=COL[m], lw=0.8, alpha=0.6)
            ax.scatter(co[ep, 0], co[ep, 1], color=COL[m], s=10, label=LAB[m])
            for i in ep:
                e = upd[i] // C.UPDATES_PER_EPOCH
                if e in (0, 6, 12, 21, 30):
                    ax.annotate(str(e), co[i, :2], fontsize=6, color=COL[m], xytext=(3, 3), textcoords="offset points")
            for t in C.transitions(m):
                w = [i for i in idx if upd[i] == t["update"]]
                if w:
                    ax.scatter(co[w, 0], co[w, 1], marker="x", color=COL[m], s=30)
        ax.set_xlabel("PC1 (%.0f%% of fit variance)" % (100 * evr[0]))
        ax.set_ylabel("PC2 (%.0f%%)" % (100 * evr[1]))
        ax.legend(frameon=False, fontsize=7)
        ax.set_title("Seed %d: checkpoints projected on one PCA basis shared by the four runs (fit on epochs 0-30).\n"
                     "Dots = epoch checkpoints (labels = epochs 0/6/12/21/30), × = transition updates; thin lines join "
                     "saved checkpoints, not observed paths.\nPC1+PC2 = %.0f%%; largest unexplained fraction of a "
                     "checkpoint's distance² = %.2f" % (s, 100 * evr[:2].sum(), frac2.max()), fontsize=7.5)
        ax.grid(alpha=0.3)
        save(fig, "T1_pca_trajectories_seed%d" % s)

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True, constrained_layout=True)
        for ax, key, lab in ((axes[0], "resid2", "2 components"), (axes[1], "resid10", "10 components")):
            for m in C.METHODS:
                idx = np.where(meth == m)[0]
                idx = idx[np.argsort(upd[idx])]
                ax.plot(upd[idx], z[key][idx], color=COL[m], lw=1, label=LAB[m])
            ax.set_xlabel("optimizer update"); ax.set_ylabel("‖projection residual‖ (L2, parameter units)")
            ax.set_title("residual after projecting on %s (seed %d)" % (lab, s)); ax.grid(alpha=0.3)
        axes[0].legend(frameon=False, fontsize=7)
        save(fig, "T3_projection_error_seed%d" % s)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True, constrained_layout=True)
        for ax, sp in zip(axes, ("train_probe", "test_probe")):
            for m in C.METHODS:
                r, _ = raw.task("traj__%s__seed%d" % (C.SHORT[m], s))
                for lab, pol, ls in (("final", "recalibrated", "-"), ("final", "saved", ":"),
                                     ("current", "recalibrated", "--"), ("current", "saved", "-.")):
                    pts = sorted([x for x in r.values() if x["label"] == lab and x["policy"] == pol],
                                 key=lambda x: x["global_update"])
                    if lab == "final":
                        pts_all = pts
                    else:
                        pts_all = pts
                    if not pts_all:
                        continue
                    ax.plot([x["global_update"] for x in pts_all], [x["splits"][sp]["ce"] for x in pts_all], ls, color=COL[m],
                            lw=0.9, label="%s: %s state, %s" % (LAB[m], lab, pol) if m == "resolution_max_b1_gaussian_conv" or lab == "final" and pol == "recalibrated" else None)
            ax.set_yscale("log"); ax.set_xlabel("optimizer update"); ax.set_ylabel("CE of the actual checkpoint")
            ax.set_title("seed %d, %s" % (s, SPL[sp])); ax.grid(alpha=0.3)
        axes[1].legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.suptitle("Actual checkpoint loss: 'final' = full resolution, no filter; 'current' = the state used by the "
                     "preceding updates (drawn only where it differs). Solid/dashed = recalibrated, dotted/dash-dot = saved BN.")
        save(fig, "T2_checkpoint_loss_seed%d" % s)
    summary["pca"] = out


# -- 6. surfaces -----------------------------------------------------------------

def grid_of(rows, sp, n):
    Z = np.full((n, n), np.nan)
    for r in rows.values():
        Z[r["j"], r["i"]] = r["splits"][sp]["ce"]
    return Z


def surface_panel(ax, g, Z, norm, cmap, zlim, title, zlabel):
    A, B = np.meshgrid(g, g)
    ax.plot_surface(A, B, Z, facecolors=cmap(norm(Z)), rstride=1, cstride=1, linewidth=0.1,
                    edgecolor=(0, 0, 0, 0.2), shade=False)
    ax.contour(A, B, Z, levels=8, zdir="z", offset=zlim[0], cmap=cmap, norm=norm, linewidths=0.6)
    c = len(g) // 2
    ax.scatter([0], [0], [Z[c, c]], color="k", s=14, depthshade=False)
    ax.set_xlim(g[0], g[-1]); ax.set_ylim(g[0], g[-1]); ax.set_zlim(*zlim)
    ax.set_box_aspect(BOX); ax.view_init(ELEV, AZIM)
    ax.set_xlabel("a (dir. 0)", labelpad=0); ax.set_ylabel("b (dir. 1)", labelpad=0)
    if zlabel:
        ax.set_zlabel(zlabel)
    ax.tick_params(pad=0, labelsize=6)
    ax.set_title(title, pad=0, fontsize=7.5)


def surfaces(raw, summary, S):
    cross, status = [], {}
    for s in C.SEEDS:
        for n in ([21, 41] if s == C.PRIMARY_SEED else [21]):
            grids = {}
            for m in C.METHODS:
                tid = "surface__%s__seed%d__g%d" % (C.SHORT[m], s, n)
                r, metas = raw.task(tid)
                status[tid] = "%s (%d/%d)" % (metas[-1]["status"] if metas else "missing", len(r), n * n)
                if len(r) == n * n:
                    grids[m] = r
            if len(grids) < 4:
                continue
            g = np.linspace(-0.25, 0.25, n)
            for centered in (False, True):
                fig = plt.figure(figsize=(14, 7.4))
                for row, sp in enumerate(("train_probe", "test_probe")):
                    Zs = {m: grid_of(grids[m], sp, n) for m in C.METHODS}
                    if centered:
                        Zs = {m: Z - Z[n // 2, n // 2] for m, Z in Zs.items()}
                    lo = min(np.nanmin(Z) for Z in Zs.values()); hi = max(np.nanmax(Z) for Z in Zs.values())
                    if centered:
                        mm = max(abs(lo), abs(hi)); norm, cmap = colors.TwoSlopeNorm(0, -mm, mm), cm.RdBu_r
                    else:
                        norm, cmap = colors.Normalize(lo, hi), cm.viridis
                    zlim = (lo - 0.05 * (hi - lo), hi)
                    for col, m in enumerate(C.METHODS):
                        ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection="3d")
                        c0 = grid_of(grids[m], sp, n)[n // 2, n // 2]
                        surface_panel(ax, g, Zs[m], norm, cmap, zlim, "%s, %s\ncentre CE %.3f" % (LAB[m], SPL[sp], c0),
                                      ("CE − centre" if centered else "CE") if col == 0 else "")
                    cax = fig.add_axes([0.93, 0.56 - row * 0.46, 0.01, 0.3])
                    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="shared in row")
                fig.suptitle("Final solutions, seed %d%s, directions 0 (a) and 1 (b), %dx%d measured grid, final state, "
                             "recalibrated BN, %s CE.\nMatched relative filter-wise perturbations (same random draws scaled "
                             "by each solution's filter norms), not one physical plane. Local slices only: a convex-looking "
                             "slice does not establish a convex neighbourhood." % (
                                 s, " (prespecified primary)" if s == C.PRIMARY_SEED else "", n, n,
                                 "centre-subtracted" if centered else "absolute"), fontsize=8.5)
                fig.subplots_adjust(left=0, right=0.91, top=0.88, bottom=0.02, wspace=0.02, hspace=0.12)
                save(fig, "A%s_surface3d_%s_seed%d_g%d" % ("2" if centered else "1", "centered" if centered else "absolute", s, n))
                # contour companion
                fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), constrained_layout=True)
                for row, sp in enumerate(("train_probe", "test_probe")):
                    Zs = {m: grid_of(grids[m], sp, n) for m in C.METHODS}
                    if centered:
                        Zs = {m: Z - Z[n // 2, n // 2] for m, Z in Zs.items()}
                    lo = min(np.nanmin(Z) for Z in Zs.values()); hi = max(np.nanmax(Z) for Z in Zs.values())
                    if centered:
                        mm = max(abs(lo), abs(hi)); norm, cmap = colors.TwoSlopeNorm(0, -mm, mm), cm.RdBu_r
                    else:
                        norm, cmap = colors.Normalize(lo, hi), cm.viridis
                    for col, m in enumerate(C.METHODS):
                        ax = axes[row, col]
                        im = ax.imshow(Zs[m], origin="lower", extent=[g[0], g[-1], g[0], g[-1]], norm=norm, cmap=cmap)
                        A_, B_ = np.meshgrid(g, g)
                        ax.contour(A_, B_, Zs[m], levels=8, colors="k", linewidths=0.4)
                        ax.plot(0, 0, "k+", ms=8)
                        ax.set_title("%s, %s" % (LAB[m], SPL[sp]), fontsize=7.5)
                        ax.set_xlabel("a (dir. 0)"); ax.set_ylabel("b (dir. 1)")
                    fig.colorbar(im, ax=axes[row, :], shrink=0.8, label="CE − centre" if centered else "CE")
                fig.suptitle("Contour companion: seed %d, %dx%d, recalibrated BN, %s" % (s, n, n, "centre-subtracted" if centered else "absolute"))
                save(fig, "A%s_contour_%s_seed%d_g%d" % ("2" if centered else "1", "centered" if centered else "absolute", s, n))
            # cross-check 1-D rows
            for m in C.METHODS:
                sens, _ = raw.task("sens1d__%s__seed%d" % (C.SHORT[m], s))
                for amp in C.AMPLITUDES:
                    for sign in (1, -1):
                        for k, axis in ((0, "a"), (1, "b")):
                            p = sens.get("recalibrated|%d|%g|%d" % (k, amp, sign))
                            i = int(np.argmin(np.abs(g - sign * amp)))
                            key = "%d|%d" % ((i, n // 2) if axis == "a" else (n // 2, i))
                            q = grids[m].get(key)
                            if p and q and abs(g[i] - sign * amp) < 1e-9:
                                cross.append(max(abs(p["splits"][sp]["ce"] - q["splits"][sp]["ce"]) for sp in ("train_probe", "test_probe")))
    summary["surface_status"] = status
    summary["crosscheck_1d_vs_2d_max_abs_ce_diff"] = max(cross) if cross else None
    summary["crosscheck_n"] = len(cross)


def fixed_surfaces(raw, summary):
    s = C.PRIMARY_SEED
    st = {}
    for m in C.METHODS[1:]:
        for tr in C.fixed_weight_transitions(m):
            states = C.fixed_weight_states(m, tr)
            grids = {}
            for lab, sd_ in states:
                tid = "fixedsurf__%s__seed%d__u%06d__%s" % (C.SHORT[m], s, tr["update"], lab.replace("=", ""))
                r, metas = raw.task(tid)
                st[tid] = "%s (%d/441)" % (metas[-1]["status"] if metas else "missing", len(r))
                if len(r) == 441:
                    grids[lab] = (sd_, r)
            if len(grids) < len(states):
                continue
            g = np.linspace(-0.25, 0.25, 21)
            for centered in (False, True):
                fig = plt.figure(figsize=(4.2 * len(states) + 1, 7.4))
                for row, sp in enumerate(("train_probe", "test_probe")):
                    Zs = {lab: grid_of(r, sp, 21) for lab, (_, r) in grids.items()}
                    if centered:
                        Zs = {k: Z - Z[10, 10] for k, Z in Zs.items()}
                    lo = min(np.nanmin(Z) for Z in Zs.values()); hi = max(np.nanmax(Z) for Z in Zs.values())
                    if centered:
                        mm = max(abs(lo), abs(hi)); norm, cmap = colors.TwoSlopeNorm(0, -mm, mm), cm.RdBu_r
                    else:
                        norm, cmap = colors.Normalize(lo, hi), cm.viridis
                    zlim = (lo - 0.05 * (hi - lo), hi)
                    for col, (lab, sd_) in enumerate(states):
                        ax = fig.add_subplot(2, len(states), row * len(states) + col + 1, projection="3d")
                        c0 = grid_of(grids[lab][1], sp, 21)[10, 10]
                        surface_panel(ax, g, Zs[lab], norm, cmap, zlim, "state %s %s\n%s, centre CE %.3f" % (lab, sd_, SPL[sp], c0),
                                      ("CE − centre" if centered else "CE") if col == 0 else "")
                    cax = fig.add_axes([0.93, 0.56 - row * 0.46, 0.01, 0.3])
                    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="shared in row")
                fig.suptitle("Fixed weights: %s, seed %d, checkpoint at update %d (weights after the last update in state %s).\n"
                             "Same weights, directions (0, 1), subsets and axes in every panel; only the intervention state "
                             "changes; recalibrated BN per state; %s CE." % (LAB[m], s, tr["update"], tr["before"],
                                                                              "centre-subtracted" if centered else "absolute"), fontsize=8.5)
                fig.subplots_adjust(left=0, right=0.91, top=0.88, bottom=0.02, wspace=0.02, hspace=0.12)
                save(fig, "B%s_fixed_surface3d_%s_%s_u%06d" % ("2" if centered else "1", "centered" if centered else "absolute",
                                                                C.SHORT[m], tr["update"]))
    summary["fixed_surface_status"] = st


def pca_plane(raw, summary):
    s = C.PRIMARY_SEED
    r, metas = raw.task("pcaplane__seed%d" % s)
    pz = raw.file("pca/pca_seed%d.npz" % s)
    summary["pca_plane_status"] = "%s (%d/%d)" % (metas[-1]["status"] if metas else "missing", len(r), C.PCA_GRID ** 2)
    if pz is None or len(r) < C.PCA_GRID ** 2:
        return
    z = np.load(pz, allow_pickle=True)
    n = C.PCA_GRID
    xs = sorted({v["x"] for v in r.values()}); ys = sorted({v["y"] for v in r.values()})
    co, meth, upd, fit = z["coords"], z["method"], z["update"], z["fit"]
    for sp in ("train_probe", "test_probe"):
        Z = np.full((n, n), np.nan)
        for v in r.values():
            Z[v["j"], v["i"]] = v["splits"][sp]["ce"]
        fig, ax = plt.subplots(figsize=(8, 6.4))
        X, Y = np.meshgrid(xs, ys)
        cf = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
        fig.colorbar(cf, ax=ax, label="CE of reconstructed in-plane points (final state, recalibrated BN)")
        for m in C.METHODS:
            idx = np.where((meth == m) & fit)[0]
            idx = idx[np.argsort(upd[idx])]
            ax.plot(co[idx, 0], co[idx, 1], "-o", color=COL[m], ms=2.5, lw=1, label=LAB[m], mec="w", mew=0.3)
            for i in idx:
                e = upd[i] // C.UPDATES_PER_EPOCH
                if e in (0, 6, 12, 21, 30):
                    ax.annotate(str(e), co[i, :2], fontsize=6, color="w", xytext=(3, 3), textcoords="offset points")
        ax.legend(frameon=False, fontsize=7, labelcolor="w")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title("Seed %d, %s. Background: loss of points reconstructed in the plane (not of the checkpoints, "
                     "which lie off it).\nEarly continuation checkpoints were trained on a different objective." % (s, SPL[sp]), fontsize=7.5)
        save(fig, "T4_pca_plane_contour_%s_seed%d" % (sp, s))
        fig = plt.figure(figsize=(8, 6.4))
        ax = fig.add_subplot(111, projection="3d")
        norm = colors.Normalize(np.nanmin(Z), np.nanmax(Z))
        fc = cm.viridis(norm(Z)); fc[..., 3] = 0.6
        zlo = np.nanmin(Z) - 0.35 * (np.nanmax(Z) - np.nanmin(Z))
        ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1, linewidth=0.1, edgecolor=(0, 0, 0, 0.2), shade=False)
        for m in C.METHODS:
            idx = np.where((meth == m) & fit)[0]
            idx = idx[np.argsort(upd[idx])]
            ax.plot(co[idx, 0], co[idx, 1], zs=zlo, zdir="z", color=COL[m], lw=1.2, label="%s (projection, floor only)" % LAB[m])
        ax.set_zlim(zlo, np.nanmax(Z)); ax.view_init(ELEV, AZIM); ax.set_box_aspect(BOX)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("CE of in-plane points")
        ax.legend(fontsize=6, frameon=False, loc="upper left")
        ax.set_title("Companion 3-D view of the same measured plane grid (seed %d, %s). Projected checkpoints are drawn "
                     "on the floor only." % (s, SPL[sp]), fontsize=7.5)
        save(fig, "T5_pca_plane_surface3d_%s_seed%d" % (sp, s))


def interactive(raw):
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception:
        return
    parts, first = [], True
    s, n = C.PRIMARY_SEED, 41
    for sp in ("train_probe", "test_probe"):
        grids = {m: raw.task("surface__%s__seed%d__g%d" % (C.SHORT[m], s, n))[0] for m in C.METHODS}
        if any(len(v) < n * n for v in grids.values()):
            n = 21
            grids = {m: raw.task("surface__%s__seed%d__g%d" % (C.SHORT[m], s, n))[0] for m in C.METHODS}
            if any(len(v) < n * n for v in grids.values()):
                return
        g = np.linspace(-0.25, 0.25, n)
        Zs = {m: grid_of(grids[m], sp, n) for m in C.METHODS}
        lo = min(Z.min() for Z in Zs.values()); hi = max(Z.max() for Z in Zs.values())
        for m in C.METHODS:
            f = go.Figure(go.Surface(x=g, y=g, z=Zs[m], cmin=lo, cmax=hi, colorscale="Viridis"))
            f.update_layout(title="%s, seed %d, %s (%dx%d, recalibrated BN, final state)" % (LAB[m], s, SPL[sp], n, n),
                            height=480, margin=dict(l=0, r=0, t=40, b=0),
                            scene=dict(zaxis=dict(range=[lo, hi]), aspectratio=dict(x=1, y=1, z=0.75),
                                       xaxis_title="a (dir 0)", yaxis_title="b (dir 1)", zaxis_title="CE"))
            parts.append(pio.to_html(f, full_html=False, include_plotlyjs="inline" if first else False))
            first = False
    html = ("<html><head><meta charset='utf-8'><title>landscape_v2 surfaces</title></head><body>"
            "<h2>landscape_v2: measured final-solution surfaces (primary seed)</h2><p>Matched relative filter-wise "
            "perturbations, not one physical plane. Shared z and colour range per split.</p>"
            + "".join("<div style='display:inline-block;width:49%%'>%s</div>" % p for p in parts) + "</body></html>")
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "surfaces_interactive.html").write_text(html, encoding="utf-8")


def inventory(raw, summary):
    inv = {}
    for d in raw.v2:
        acc = d.parent.name
        rec = {}
        for f in ("timing.json", "environment.json", "pca/pca_summary.json"):
            if (d / f).exists():
                rec[f] = json.loads((d / f).read_text())
        rec["checks"] = {p.name: json.loads(p.read_text())["summary"] for p in sorted((d / "checks").glob("*.json"))}
        v = C.STUDY / "raw" / ("verify_%s.json" % acc)
        rec["download_verification"] = json.loads(v.read_text()) if v.exists() else "not run"
        metas = [json.loads(p.read_text()) for p in sorted((d / "eval").glob("*.meta.json"))]
        rec["task_status"] = {m["task"]["id"]: m["status"] for m in metas}
        rec["failed_tasks"] = {m["task"]["id"]: m.get("traceback", "")[-800:] for m in metas if m["status"] == "failed"}
        runs = {}
        for rd in sorted((d / "runs").glob("*__seed*")):
            sm = rd / "summary.json"
            if sm.exists():
                j = json.loads(sm.read_text())
                runs[rd.name] = {"final_test_acc": j["final"]["target"]["test"]["acc"],
                                 "final_test_ce": j["final"]["target"]["test"]["ce"],
                                 "wall_seconds": j["timing"]["wall_seconds"],
                                 "n_checkpoints": len(list((rd / "checkpoints").glob("*.pt")))}
            else:
                runs[rd.name] = "incomplete (no summary.json)"
        rec["runs"] = runs
        inv[acc] = rec
    summary["inventory"] = inv
    acc_rows = []
    for acc, rec in inv.items():
        for run, v in rec["runs"].items():
            if isinstance(v, dict):
                m, s = run.split("__seed")
                acc_rows.append({"account": acc, "method": m, "seed": int(s), **v})
    write_csv("training_runs", sorted(acc_rows, key=lambda r: (r["seed"], C.METHODS.index(r["method"]))))
    summary["issues"] = raw.issues


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(C.STUDY / "raw"))
    ap.add_argument("--out", default=None, help="write figures/ and tables/ here instead of the study dir")
    args = ap.parse_args()
    if args.out:
        global FIG, TAB
        FIG, TAB = Path(args.out) / "figures", Path(args.out) / "tables"
    raw = Raw(Path(args.raw))
    summary = {}
    inventory(raw, summary)
    S = sensitivity(raw, summary)
    for f in (validation,):
        f(raw, S, summary)
    for f in (interpolation, fixed_weight, trajectories, fixed_surfaces, pca_plane):
        try:
            f(raw, summary)
        except Exception as exc:
            import traceback
            summary.setdefault("analysis_errors", {})[f.__name__] = traceback.format_exc()
    try:
        surfaces(raw, summary, S)
    except Exception:
        import traceback
        summary.setdefault("analysis_errors", {})["surfaces"] = traceback.format_exc()
    interactive(raw)
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("analysis written:", FIG, TAB, "errors:", list(summary.get("analysis_errors", {})))


if __name__ == "__main__":
    main()

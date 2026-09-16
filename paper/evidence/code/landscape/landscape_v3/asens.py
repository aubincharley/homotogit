"""Analysis of blocks A (amplitudes, float64), B (validation) and C (temporal, transitions)."""
from __future__ import annotations

import json

import numpy as np
import torch

from . import common as C
from .acommon import (COL, LAB, POL, SPL, S_of, mark_transitions, plt, save, seedstats, state_label,
                      write_csv)

V2 = C.V2


# -- A. final-solution amplitudes ------------------------------------------------------

def amplitudes(D, summary, quad):
    per_dir, per_seed, centres, missing = [], [], [], 0
    Sbar = {}
    for m in C.METHODS:
        tgt = C.target(m)
        for s in C.SEEDS:
            for pol in C.POLICIES:
                for sp in C.PROBES:
                    ck = C.pkey(m, s, C.FINAL, tgt, pol, "probes", "c")
                    if D.ce(ck, sp) is not None:
                        centres.append({"method": m, "seed": s, "policy": pol, "split": sp,
                                        "centre_ce": D.ce(ck, sp), "centre_acc": D.acc(ck, sp)})
                    for amp in C.AMPLITUDES:
                        vals, cs = [], []
                        for k in range(C.N_DIRECTIONS):
                            r = S_of(D, m, s, C.FINAL, tgt, pol, "probes", sp, k, amp)
                            if r is None:
                                missing += 1
                                continue
                            Cd = 2 * r["S"] / amp ** 2
                            vals.append(r["S"])
                            cs.append(Cd)
                            per_dir.append({"method": m, "seed": s, "policy": pol, "split": sp, "direction": k,
                                            "amplitude": amp, "L_centre": r["L0"], "dL_plus": r["Lp"] - r["L0"],
                                            "dL_minus": r["Lm"] - r["L0"], "S": r["S"], "C_d": Cd})
                        if vals:
                            Sbar[(m, s, pol, sp, amp)] = float(np.mean(vals))
                            per_seed.append({"method": m, "seed": s, "policy": pol, "split": sp, "amplitude": amp,
                                             "n_directions": len(vals), "S_mean": float(np.mean(vals)),
                                             "S_direction_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                                             "S_min": float(np.min(vals)), "S_max": float(np.max(vals)),
                                             "n_negative_S": int(sum(v < 0 for v in vals)),
                                             "C_mean": float(np.mean(cs)), "C_direction_sd": float(np.std(cs, ddof=1))
                                             if len(cs) > 1 else float("nan")})
    write_csv("A_sensitivity_per_direction", per_dir)
    write_csv("A_sensitivity_per_seed", per_seed)
    write_csv("A_centres", centres)
    paired = []
    for pol in C.POLICIES:
        for sp in C.PROBES:
            for amp in C.AMPLITUDES:
                for m in C.METHODS[1:]:
                    d = {s: Sbar[(m, s, pol, sp, amp)] - Sbar[("plain", s, pol, sp, amp)] for s in C.SEEDS
                         if (m, s, pol, sp, amp) in Sbar and ("plain", s, pol, sp, amp) in Sbar}
                    rel = {s: d[s] / Sbar[("plain", s, pol, sp, amp)] for s in d}
                    row = {"policy": pol, "split": sp, "amplitude": amp, "method": m, **seedstats(d)}
                    row["mean_relative_to_plain"] = float(np.mean(list(rel.values()))) if rel else float("nan")
                    paired.append(row)
    write_csv("A_paired_vs_plain", paired)
    cent_paired = []
    for pol in C.POLICIES:
        for sp in C.PROBES:
            for m in C.METHODS[1:]:
                vals = {}
                for s in C.SEEDS:
                    a = [r["centre_ce"] for r in centres if r["method"] == m and r["seed"] == s and r["policy"] == pol and r["split"] == sp]
                    b = [r["centre_ce"] for r in centres if r["method"] == "plain" and r["seed"] == s and r["policy"] == pol and r["split"] == sp]
                    if a and b:
                        vals[s] = a[0] - b[0]
                cent_paired.append({"policy": pol, "split": sp, "method": m, "quantity": "centre CE - plain", **seedstats(vals)})
    write_csv("A_centre_paired_vs_plain", cent_paired)
    summary["A"] = {"missing_direction_points": missing, "paired": paired, "centre_paired": cent_paired}

    # float64 check
    f64 = []
    for m in C.METHODS:
        tgt = C.target(m)
        for s in C.SEEDS:
            for pol in C.POLICIES:
                for sp in C.PROBES:
                    for k in C.F64_CHECK["directions"]:
                        for amp in C.F64_CHECK["amplitudes"]:
                            a = S_of(D, m, s, C.FINAL, tgt, pol, "probes", sp, k, amp, "f32")
                            b = S_of(D, m, s, C.FINAL, tgt, pol, "probes", sp, k, amp, "f64")
                            if a and b:
                                diff = a["S"] - b["S"]
                                f64.append({"method": m, "seed": s, "policy": pol, "split": sp, "direction": k,
                                            "amplitude": amp, "S_f32": a["S"], "S_f64": b["S"], "diff": diff,
                                            "L0_diff": a["L0"] - b["L0"],
                                            "suspicious": abs(diff) > 1e-6 + 1e-3 * abs(b["S"]),
                                            "C_f32": 2 * a["S"] / amp ** 2, "C_f64": 2 * b["S"] / amp ** 2})
    write_csv("A_float64_check", f64)
    summary["A_float64"] = {"n": len(f64), "n_suspicious": int(sum(r["suspicious"] for r in f64)),
                            "max_abs_diff_S": max((abs(r["diff"]) for r in f64), default=float("nan")),
                            "max_rel_diff_S": max((abs(r["diff"]) / max(abs(r["S_f64"]), 1e-30) for r in f64), default=float("nan")),
                            "n_negative_S_f64": int(sum(r["S_f64"] < 0 for r in f64))}
    figures_A(Sbar, paired, centres, per_dir, per_seed, f64, quad)
    return Sbar


def figures_A(Sbar, paired, centres, per_dir, per_seed, f64, quad):
    amps = np.array(C.AMPLITUDES)
    # A1 centre CE (separate from sensitivity)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True, sharey=True)
    for ax, sp in zip(axes, C.PROBES):
        for j, pol in enumerate(C.POLICIES):
            for i, m in enumerate(C.METHODS):
                v = [r["centre_ce"] for r in centres if r["method"] == m and r["policy"] == pol and r["split"] == sp]
                x = j * 5 + i
                ax.scatter([x] * len(v), v, color=COL[m], s=12, label=LAB[m] if j == 0 else None)
                if v:
                    ax.plot([x - 0.3, x + 0.3], [np.mean(v)] * 2, color="k", lw=1.2)
        ax.set_xticks([1.5, 6.5, 11.5], [POL[p] for p in C.POLICIES], fontsize=7)
        ax.set_title(SPL[sp])
        ax.set_ylabel("centre CE (unperturbed loss)")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Centre CE of the epoch-30 networks, final state")
    save(fig, "A1_centre_ce")

    # A2 S(eps) and A3 paired differences
    for name in ("A2_sensitivity_vs_amplitude", "A3_paired_difference_vs_amplitude"):
        fig, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True, sharex=True)
        for i, pol in enumerate(C.POLICIES):
            for j, sp in enumerate(C.PROBES):
                ax = axes[i, j]
                if name.startswith("A2"):
                    allpos = True
                    for m in C.METHODS:
                        mu, sg = [], []
                        for amp in C.AMPLITUDES:
                            v = [Sbar[(m, s, pol, sp, amp)] for s in C.SEEDS if (m, s, pol, sp, amp) in Sbar]
                            allpos &= all(x > 0 for x in v)
                            ax.scatter([amp] * len(v), v, color=COL[m], s=5, alpha=0.4)
                            mu.append(np.mean(v) if v else np.nan)
                            sg.append(np.std(v, ddof=1) if len(v) > 1 else np.nan)
                        ax.errorbar(amps, mu, yerr=sg, color=COL[m], marker="o", ms=2.5, lw=1.1, capsize=2, label=LAB[m])
                    ax.set_xscale("log")
                    if allpos:
                        ax.set_yscale("log")
                    else:
                        ax.set_yscale("symlog", linthresh=1e-5)
                    ax.set_ylabel("S̄(ε), mean of 20 directions")
                else:
                    for mi, m in enumerate(C.METHODS[1:]):
                        for amp in C.AMPLITUDES:
                            row = [r for r in paired if r["policy"] == pol and r["split"] == sp and r["amplitude"] == amp and r["method"] == m][0]
                            x = amp * (1.12 ** (mi - 1))
                            v = [row["seed%d" % s] for s in C.SEEDS]
                            ax.scatter([x] * 5, v, color=COL[m], s=8, alpha=0.8, label=LAB[m] if amp == C.AMPLITUDES[0] else None)
                            ax.plot([x / 1.04, x * 1.04], [row["mean"]] * 2, color=COL[m], lw=2)
                    ax.axhline(0, color="k", lw=0.6)
                    ax.set_xscale("log")
                    ax.set_yscale("symlog", linthresh=1e-3)
                    ax.set_ylabel("Δ = S̄(method) − S̄(plain), same seed")
                ax.set_title("%s, %s" % (POL[pol], SPL[sp]))
                ax.grid(alpha=0.3)
                if i == 2:
                    ax.set_xlabel("ε (RMS relative block displacement)")
        axes[0, 0].legend(frameon=False, fontsize=7)
        fig.suptitle("Final solutions: symmetric sensitivity by amplitude" if name.startswith("A2")
                     else "Final solutions: paired differences against plain (below 0 = smaller increase)")
        save(fig, name)

    # A4 curvature proxy
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True, sharex=True)
    for i, pol in enumerate(C.POLICIES):
        for j, sp in enumerate(C.PROBES):
            ax = axes[i, j]
            for m in C.METHODS:
                mu = []
                for amp in C.AMPLITUDES:
                    v = [r["C_mean"] for r in per_seed if r["method"] == m and r["policy"] == pol and r["split"] == sp and r["amplitude"] == amp]
                    ax.scatter([amp] * len(v), v, color=COL[m], s=5, alpha=0.4)
                    mu.append(np.mean(v) if v else np.nan)
                ax.plot(amps, mu, color=COL[m], marker="o", ms=2.5, lw=1.1, label=LAB[m])
                if pol in (C.SAVED, C.CFROZEN) and quad:
                    q = [quad.get((m, s, sp, pol)) for s in C.SEEDS]
                    q = [x for x in q if x is not None]
                    if q:
                        ax.scatter([0.0035] * len(q), q, color=COL[m], marker="_", s=60)
                        ax.axhline(np.mean(q), color=COL[m], ls="--", lw=0.7)
            ax.set_xscale("log")
            ax.set_xlim(0.003, 0.6)
            ax.set_title("%s, %s" % (POL[pol], SPL[sp]))
            ax.set_ylabel("C̄(ε) = mean 2S/ε²")
            ax.grid(alpha=0.3)
            if i == 2:
                ax.set_xlabel("ε")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Finite-difference curvature proxy; dashes at left and dashed lines = mean d·H·d from HVPs (frozen BN only)")
    save(fig, "A4_curvature_proxy")

    # A5 direction spread
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for i, pol in enumerate(C.POLICIES):
        for j, amp in enumerate((0.01, 0.1, 0.5)):
            ax = axes[i, j]
            for mi, m in enumerate(C.METHODS):
                for s in C.SEEDS:
                    v = [r["S"] for r in per_dir if r["method"] == m and r["seed"] == s and r["policy"] == pol
                         and r["split"] == "test_probe" and r["amplitude"] == amp]
                    if v:
                        ax.boxplot(v, positions=[mi * 6 + s], widths=0.7, patch_artist=True,
                                   boxprops=dict(facecolor=COL[m], alpha=0.4), medianprops=dict(color="k"),
                                   flierprops=dict(markersize=2))
            ax.set_xticks([mi * 6 + 2 for mi in range(4)], [LAB[m] for m in C.METHODS], fontsize=7)
            ax.set_yscale("symlog", linthresh=1e-4)
            ax.set_title("%s, ε = %g" % (POL[pol], amp))
            ax.set_ylabel("S, one direction")
            ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Direction-to-direction spread on the test probe (boxes: 20 directions for each of seeds 0-4)")
    save(fig, "A5_direction_spread")

    # A6 float64
    if f64:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        for pol, mk in zip(C.POLICIES, ("o", "^", "s")):
            r = [x for x in f64 if x["policy"] == pol]
            axes[0].scatter([x["S_f64"] for x in r], [x["S_f32"] for x in r], s=8, marker=mk, alpha=0.6, label=POL[pol])
            axes[1].scatter([x["amplitude"] for x in r], [abs(x["diff"]) + 1e-12 for x in r], s=8, marker=mk, alpha=0.6)
        lo = min(x["S_f64"] for x in f64)
        hi = max(x["S_f64"] for x in f64)
        axes[0].plot([lo, hi], [lo, hi], "k:", lw=0.8)
        axes[0].set_xscale("symlog", linthresh=1e-5); axes[0].set_yscale("symlog", linthresh=1e-5)
        axes[0].set_xlabel("S, float64"); axes[0].set_ylabel("S, float32"); axes[0].legend(frameon=False, fontsize=7)
        axes[1].set_xscale("log"); axes[1].set_yscale("log")
        axes[1].set_xlabel("ε"); axes[1].set_ylabel("|S_f32 − S_f64| (+1e-12)")
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.suptitle("Precision check: directions 0-1, both probes, all networks")
        save(fig, "A6_float64_check")


# -- B. validation ---------------------------------------------------------------------

def validation(D, summary, Sbar):
    rows, per_seed = [], {}
    pairs = (("train_large", "train_probe"), ("test_full", "test_probe"))
    for m in C.METHODS:
        tgt = C.target(m)
        for s in C.SEEDS:
            for pol in (C.SAVED, C.POINTWISE):
                c = C.pkey(m, s, C.FINAL, tgt, pol, "large", "c")
                for big, small in pairs:
                    for amp in C.VALIDATION_AMPS:
                        vb, vs = [], []
                        for k in range(C.N_DIRECTIONS):
                            a = S_of(D, m, s, C.FINAL, tgt, pol, "large", big, k, amp)
                            b = S_of(D, m, s, C.FINAL, tgt, pol, "probes", small, k, amp)
                            if a and b:
                                vb.append(a["S"])
                                vs.append(b["S"])
                        if vb:
                            per_seed[(m, s, pol, big, amp)] = (float(np.mean(vb)), float(np.mean(vs)), len(vb))
                            rows.append({"method": m, "seed": s, "policy": pol, "set": big, "probe": small, "amplitude": amp,
                                         "n_directions": len(vb), "S_large": float(np.mean(vb)), "S_probe_same_dirs": float(np.mean(vs)),
                                         "centre_ce_large": D.ce(c, big), "centre_acc_large": D.acc(c, big)})
    write_csv("B_validation_per_seed", rows)
    out = []
    for pol in (C.SAVED, C.POINTWISE):
        for big, small in pairs:
            for amp in C.VALIDATION_AMPS:
                for m in C.METHODS[1:]:
                    dl, dp = {}, {}
                    for s in C.SEEDS:
                        a, b = per_seed.get((m, s, pol, big, amp)), per_seed.get(("plain", s, pol, big, amp))
                        if a and b and a[2] == b[2] == C.N_DIRECTIONS:
                            dl[s], dp[s] = a[0] - b[0], a[1] - b[1]
                    L, P = seedstats(dl), seedstats(dp)
                    out.append({"policy": pol, "set": big, "probe": small, "amplitude": amp, "method": m,
                                "n_seeds": L["n_seeds"], "mean_diff_large": L["mean"], "sd_large": L["sd"],
                                "n_below_plain_large": L["n_negative"], "n_above_plain_large": L["n_positive"],
                                "mean_diff_probe": P["mean"], "sd_probe": P["sd"], "n_below_plain_probe": P["n_negative"],
                                "n_same_sign_probe_vs_large": int(sum(np.sign(dl[s]) == np.sign(dp[s]) for s in dl)),
                                "ratio_mean_large_over_probe": L["mean"] / P["mean"] if P["mean"] else float("nan"),
                                **{"seed%d_large" % s: dl.get(s, float("nan")) for s in C.SEEDS},
                                **{"seed%d_probe" % s: dp.get(s, float("nan")) for s in C.SEEDS}})
    write_csv("B_validation_paired", out)
    summary["B"] = out
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), constrained_layout=True)
    for i, pol in enumerate((C.SAVED, C.POINTWISE)):
        for j, (big, small) in enumerate(pairs):
            for jj, amp in enumerate(C.VALIDATION_AMPS):
                ax = axes[i, 2 * j + jj]
                for m in C.METHODS[1:]:
                    row = [r for r in out if r["policy"] == pol and r["set"] == big and r["amplitude"] == amp and r["method"] == m][0]
                    xs = [row["seed%d_probe" % s] for s in C.SEEDS]
                    ys = [row["seed%d_large" % s] for s in C.SEEDS]
                    ax.scatter(xs, ys, color=COL[m], s=14, label=LAB[m])
                lim = np.nanmax(np.abs(np.concatenate([ax.get_xlim(), ax.get_ylim()])))
                ax.plot([-lim, lim], [-lim, lim], "k:", lw=0.7)
                ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
                ax.set_title("%s\n%s, ε = %g" % (POL[pol], SPL[big], amp), fontsize=7.5)
                ax.set_xlabel("Δ on the 1,000-image probe"); ax.set_ylabel("Δ on the larger set")
                ax.grid(alpha=0.3)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Paired differences against plain (20 directions, per seed): probe vs larger evaluation set")
    save(fig, "B1_validation_probe_vs_large")


# -- C. temporal -----------------------------------------------------------------------

def weight_norms():
    from continuation_core.models import build_model
    from landscape_study.directions import in_mask
    names = [n for n, p in build_model("resnet20_bn_cifar", 10).named_parameters()]
    rows = []
    for m in C.METHODS:
        for s in C.SEEDS:
            for e in C.TEMPORAL_EPOCHS:
                st = torch.load(C.v2_run_dir(m, s) / "checkpoints" / C.epoch_file(e), map_location="cpu", weights_only=False)["model_state"]
                w = [st[n].double() for n in names if in_mask(n, st[n])]
                bn = torch.cat([t.reshape(t.shape[0], -1).norm(dim=1) for t in w])
                rows.append({"method": m, "seed": s, "epoch": e,
                             "masked_weight_norm": float(torch.sqrt(sum((t ** 2).sum() for t in w))),
                             "mean_block_norm": float(bn.mean()), "fc_weight_norm": float(st["fc.weight"].double().norm()),
                             "bn_gamma_norm": float(torch.sqrt(sum((st[n].double() ** 2).sum() for n in names
                                                                   if n.endswith("weight") and st[n].dim() == 1)))})
    return rows


def produced(m, s, e):
    ck = torch.load(C.v2_run_dir(m, s) / "checkpoints" / C.epoch_file(e), map_location="cpu", weights_only=False)
    u = ck["intervention"]["used_for_last_update"]["state"]
    n = ck["intervention"]["next_update"]
    return ({"resolution": u["resolution"], "sigma": u["sigma"]},
            None if n is None else {"resolution": n["state"]["resolution"], "sigma": n["state"]["sigma"]})


def temporal(D, summary):
    rows = []
    for m in C.METHODS:
        tgt = C.target(m)
        for s in C.SEEDS:
            for e in C.TEMPORAL_EPOCHS:
                used, nxt = produced(m, s, e)
                states = [("produced", used)] if m == "plain" else [("produced", used), ("final", tgt)]
                for lab, st in states:
                    if m == "plain":
                        st = tgt
                    for pol in (C.SAVED, C.POINTWISE):
                        for sp in C.PROBES:
                            rec = {"method": m, "seed": s, "epoch": e, "update": e * C.UPE, "state_label": lab,
                                   "state": state_label(st), "produced_state": state_label(used),
                                   "next_update_state": state_label(nxt), "equals_final": st == tgt,
                                   "policy": pol, "split": sp,
                                   "centre_ce": D.ce(C.pkey(m, s, C.epoch_file(e), st, pol, "probes", "c"), sp),
                                   "centre_acc": D.acc(C.pkey(m, s, C.epoch_file(e), st, pol, "probes", "c"), sp)}
                            for amp in C.TEMPORAL_AMPS:
                                v = [S_of(D, m, s, C.epoch_file(e), st, pol, "probes", sp, k, amp) for k in C.TEMPORAL_DIRS]
                                v = [x["S"] for x in v if x]
                                rec["S_eps%g" % amp] = float(np.mean(v)) if len(v) == len(C.TEMPORAL_DIRS) else float("nan")
                                rec["n_dirs_eps%g" % amp] = len(v)
                            rows.append(rec)
    norms = weight_norms()
    write_csv("C_temporal_per_seed", rows)
    write_csv("C_weight_norms", norms)
    # paired vs plain per epoch (same seed, same epoch)
    paired = []
    for pol in (C.SAVED, C.POINTWISE):
        for sp in C.PROBES:
            for lab in ("produced", "final"):
                for m in C.METHODS[1:]:
                    for e in C.TEMPORAL_EPOCHS:
                        for q in ("centre_ce", "S_eps0.1", "S_eps0.25"):
                            vals = {}
                            for s in C.SEEDS:
                                a = [r[q] for r in rows if r["method"] == m and r["seed"] == s and r["epoch"] == e and r["policy"] == pol and r["split"] == sp and (r["state_label"] == lab or (lab == "final" and r["equals_final"]))]
                                b = [r[q] for r in rows if r["method"] == "plain" and r["seed"] == s and r["epoch"] == e and r["policy"] == pol and r["split"] == sp]
                                if a and b and a[0] is not None and b[0] is not None and np.isfinite(a[0]) and np.isfinite(b[0]):
                                    vals[s] = a[0] - b[0]
                            paired.append({"policy": pol, "split": sp, "method_state": lab, "method": m, "epoch": e,
                                           "quantity": q, **seedstats(vals)})
    write_csv("C_temporal_paired_vs_plain", paired)
    summary["C_temporal_paired"] = paired
    figures_C(rows, norms, paired)
    return rows


def figures_C(rows, norms, paired):
    E = list(C.TEMPORAL_EPOCHS)
    for sp in C.PROBES:
        fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True, sharex=True)
        for j, pol in enumerate((C.SAVED, C.POINTWISE)):
            for i, (q, ylab) in enumerate((("centre_ce", "centre CE"), ("S_eps0.1", "S̄(0.1), 10 directions"),
                                           ("S_eps0.25", "S̄(0.25), 10 directions"))):
                ax = axes[i, j]
                for m in C.METHODS:
                    for lab, ls in (("produced", "-"), ("final", "--")):
                        if m == "plain" and lab == "final":
                            continue
                        mu = []
                        for e in E:
                            v = [r[q] for r in rows if r["method"] == m and r["epoch"] == e and r["policy"] == pol and r["split"] == sp
                                 and (r["state_label"] == lab or (lab == "final" and r["equals_final"] and r["state_label"] == "produced" and m != "plain"))]
                            v = [x for x in v if x is not None and np.isfinite(x)]
                            ax.scatter([e + (0.15 if lab == "final" else 0)] * len(v), v, color=COL[m], s=4, alpha=0.35)
                            mu.append(np.mean(v) if v else np.nan)
                        ax.plot(E, mu, ls, color=COL[m], lw=1.2, marker="o", ms=2.5,
                                label=("%s, %s state" % (LAB[m], "training (produced)" if lab == "produced" else "final")) if i == 0 and j == 0 else None)
                mark_transitions(ax)
                ax.set_yscale("log")
                ax.set_ylabel(ylab)
                ax.set_title("%s" % POL[pol])
                ax.grid(alpha=0.25)
                if i == 2:
                    ax.set_xlabel("epoch (checkpoint after that many epochs)")
        axes[0, 0].legend(frameon=False, fontsize=6.5, ncol=2)
        fig.suptitle("Training-time checkpoints, %s: centre CE and sensitivity (relative directions renormalised at each checkpoint)" % SPL[sp])
        save(fig, "C1_temporal_%s" % sp)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    for ax, (q, ylab) in zip(axes, (("masked_weight_norm", "‖conv + fc weights‖₂"), ("mean_block_norm", "mean filter/row norm"),
                                    ("bn_gamma_norm", "‖BN scale parameters‖₂"))):
        for m in C.METHODS:
            mu = [np.mean([r[q] for r in norms if r["method"] == m and r["epoch"] == e]) for e in E]
            for s in C.SEEDS:
                ax.plot(E, [[r[q] for r in norms if r["method"] == m and r["epoch"] == e and r["seed"] == s][0] for e in E],
                        color=COL[m], lw=0.4, alpha=0.4)
            ax.plot(E, mu, color=COL[m], lw=1.4, label=LAB[m])
        mark_transitions(ax)
        ax.set_xlabel("epoch"); ax.set_ylabel(ylab); ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Weight norms at the temporal checkpoints (thin lines = seeds)")
    save(fig, "C2_weight_norms")
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), constrained_layout=True, sharex=True)
    for i, pol in enumerate((C.SAVED, C.POINTWISE)):
        for j, (sp, q) in enumerate([(sp, q) for sp in C.PROBES for q in ("S_eps0.1", "S_eps0.25")]):
            ax = axes[i, j]
            for m in C.METHODS[1:]:
                for lab, ls in (("produced", "-"), ("final", "--")):
                    rr = [r for r in paired if r["policy"] == pol and r["split"] == sp and r["method_state"] == lab and r["method"] == m and r["quantity"] == q]
                    rr.sort(key=lambda r: r["epoch"])
                    ax.plot([r["epoch"] for r in rr], [r["mean"] for r in rr], ls, color=COL[m], lw=1.1,
                            label=("%s, %s" % (LAB[m], lab)) if i == 0 and j == 0 else None)
                    for r in rr:
                        ax.scatter([r["epoch"]] * 5, [r["seed%d" % s] for s in C.SEEDS], color=COL[m], s=3, alpha=0.35)
            ax.axhline(0, color="k", lw=0.6)
            mark_transitions(ax)
            ax.set_yscale("symlog", linthresh=1e-2)
            ax.set_title("%s, %s, %s" % (POL[pol], SPL[sp], q.replace("S_eps", "ε = ")), fontsize=7.5)
            ax.set_ylabel("S̄(method) − S̄(plain), same epoch and seed"); ax.grid(alpha=0.25)
            if i == 1:
                ax.set_xlabel("epoch")
    axes[0, 0].legend(frameon=False, fontsize=6)
    fig.suptitle("When do the sensitivity differences appear? Paired against plain at the same epoch (dots = seeds)")
    save(fig, "C3_temporal_paired")


def transitions(D, summary):
    rows = []
    for m in C.METHODS[1:]:
        for tr in V2.transitions(m):
            e = tr["update"] // C.UPE
            joint = (tr["before"].get("resolution") != tr["after"].get("resolution")
                     and tr["before"].get("sigma") != tr["after"].get("sigma"))
            for lab, st in V2.fixed_weight_states(m, tr):
                for s in C.SEEDS:
                    for sp in C.PROBES:
                        rec = {"method": m, "epoch": e, "update": tr["update"], "joint_change": joint, "state_label": lab,
                               "state": state_label(st), "seed": s, "split": sp}
                        for pol in (C.SAVED, C.POINTWISE):
                            ck = C.pkey(m, s, C.epoch_file(e), st, pol, "probes", "c")
                            rec["centre_ce_" + pol] = D.ce(ck, sp)
                            rec["centre_acc_" + pol] = D.acc(ck, sp)
                        for pol, amps in ((C.POINTWISE, C.TRANSITION_AMPS_RECAL), (C.SAVED, C.TRANSITION_AMPS_SAVED)):
                            for amp in amps:
                                v = [S_of(D, m, s, C.epoch_file(e), st, pol, "probes", sp, k, amp) for k in C.TRANSITION_DIRS]
                                v = [x["S"] for x in v if x]
                                rec["S_%s_eps%g" % (pol, amp)] = float(np.mean(v)) if len(v) == len(C.TRANSITION_DIRS) else float("nan")
                        rows.append(rec)
    write_csv("C_transitions_per_seed", rows)
    # audits: is the trained ("before") state the lowest centre CE?  does "after" raise or lower CE?
    audit = []
    for m in C.METHODS[1:]:
        for tr in V2.transitions(m):
            e = tr["update"] // C.UPE
            for sp in C.PROBES:
                for pol in (C.SAVED, C.POINTWISE):
                    n_lowest_before, n_cases, n_after_lower = 0, 0, 0
                    for s in C.SEEDS:
                        rr = {r["state_label"]: r["centre_ce_" + pol] for r in rows if r["method"] == m and r["epoch"] == e and r["seed"] == s and r["split"] == sp}
                        if "before" in rr and all(v is not None for v in rr.values()):
                            n_cases += 1
                            n_lowest_before += int(rr["before"] == min(rr.values()))
                            n_after_lower += int(rr["after"] < rr["before"])
                    audit.append({"method": m, "epoch": e, "split": sp, "policy": pol, "n_seeds": n_cases,
                                  "n_before_state_lowest": n_lowest_before, "n_after_lower_than_before": n_after_lower})
    write_csv("C_transitions_audit", audit)
    summary["C_transitions_audit"] = audit
    methods = C.METHODS[1:]
    fig, axes = plt.subplots(3, 3, figsize=(14, 9.5), constrained_layout=True)
    for j, m in enumerate(methods):
        trs = V2.transitions(m)
        for i, (q, ylab) in enumerate((("centre_ce_recalibrated", "centre CE (recalibrated at every point)"),
                                       ("S_recalibrated_eps0.1", "S̄(0.1), recalibrated at every point"),
                                       ("S_saved_eps0.1", "S̄(0.1), saved/frozen BN"))):
            ax = axes[i, j]
            xt, xl = [], []
            for ti, tr in enumerate(trs):
                e = tr["update"] // C.UPE
                states = [lab for lab, _ in V2.fixed_weight_states(m, tr)]
                for si, lab in enumerate(states):
                    v = [r[q] for r in rows if r["method"] == m and r["epoch"] == e and r["state_label"] == lab and r["split"] == "test_probe"]
                    v = [x for x in v if x is not None and np.isfinite(x)]
                    x = ti * 4 + si
                    mk = {"before": "o", "after": "^", "final": "s"}.get(lab, "D")
                    ax.scatter([x] * len(v), v, color=COL[m], s=10, marker=mk, alpha=0.7,
                               label=lab if (ti == 0 and i == 0) else None)
                    if v:
                        ax.plot([x - 0.3, x + 0.3], [np.mean(v)] * 2, color="k", lw=1)
                joint = (tr["before"].get("resolution") != tr["after"].get("resolution") and tr["before"].get("sigma") != tr["after"].get("sigma"))
                xt.append(ti * 4 + 1)
                xl.append("ep %d%s" % (e, "\njoint r+G" if joint else ""))
            ax.set_xticks(xt, xl, fontsize=6.5)
            ax.set_yscale("log")
            ax.set_ylabel(ylab, fontsize=7)
            ax.grid(alpha=0.25, axis="y")
            if i == 0:
                ax.set_title(LAB[m])
                ax.legend(frameon=False, fontsize=6)
    fig.suptitle("Fixed weights at every schedule transition (test probe): before / after / final state; dots = seeds, bar = mean")
    save(fig, "C4_transitions")
    return rows

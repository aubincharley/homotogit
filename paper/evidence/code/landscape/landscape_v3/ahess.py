"""Analysis of the Hessian block: spectra, convergence, verification, quadratic forms, cuts."""
from __future__ import annotations

import numpy as np
import torch

from . import common as C
from .acommon import COL, LAB, POL, SPL, plt, save, seedstats, write_csv

TARGETS = ("top1", "top2", "min")


def parse_pid(pid):
    short, s, probe, rest = pid.split("_", 3)
    coords = rest[-3:]
    pol = rest[:-4]
    return C.LONG[short], int(s[1:]), probe + "_probe", pol, {"ord": "ordinary", "rel": "relative"}[coords]


def spectra(D, summary):
    H = D.hessian()
    rows = []
    for pid, h in sorted(H.items()):
        m, s, probe, pol, coords = parse_pid(pid)
        st0 = h["starts"]["0"]
        rec = {"problem": pid, "method": m, "seed": s, "probe": probe, "policy": pol, "coords": coords,
               "loss": h["loss"], "grad_norm": h["grad_norm"], "grad_norm_relative_coords": h["grad_norm_relative_coords"],
               "n_params": h["n_params"], "n_nonzero_blocks": h["block_info"]["n_nonzero_blocks"],
               "n_zero_norm_blocks": h["block_info"]["n_zero_norm_blocks"],
               "iterations_start0": st0["iterations"], "orthogonality_error_start0": st0["orthogonality_error"],
               "seconds_start0": st0["seconds"]}
        for t in TARGETS:
            rs = st0["residuals"][t]
            conv = bool(st0["converged_explicit"][t])
            rec["%s_ritz" % t] = rs["ritz"]
            rec["%s_rel_resid" % t] = rs["rel_resid"]
            rec["%s_converged" % t] = conv
            rec["%s" % t] = rs["ritz"] if conv else float("nan")
            if "1" in h["starts"]:
                s1 = h["starts"]["1"]
                rec["%s_start1_ritz" % t] = s1["residuals"][t]["ritz"]
                rec["%s_start1_converged" % t] = bool(s1["converged_explicit"][t])
                rec["%s_start_rel_diff" % t] = h["start_agreement_rel"][t]
            if t != "top2":
                v = h["verification"][t]
                rec["%s_rq_f32" % t] = v["rayleigh_f32_coordinates"]
                rec["%s_rq_f64" % t] = v["rayleigh_f64_coordinates"]
                rec["%s_quad_r1_f64" % t] = v["quadratic_f64_r_normalised"]
                rec["%s_r_of_direction" % t] = v["r_of_mapped_direction"]
                rec["%s_grad_dot_dir_r1" % t] = v["gradient_dot_direction_r_normalised"]
                for tt, fd in v["finite_differences"].items():
                    rec["%s_Cfd_t%s" % (t, tt)] = fd["C_fd"]
        rec["iterations_start1"] = h["starts"].get("1", {}).get("iterations")
        rec["min_negative_verified"] = bool(rec["min_converged"] and rec["min_rq_f32"] < 0 and rec["min_rq_f64"] < 0)
        rows.append(rec)
    write_csv("D_hessian_problems", rows)
    agg = []
    for coords in C.HESS_COORDS:
        for pol in C.HESS_POLICIES:
            for probe in C.HESS_PROBES:
                for q in ("top1", "top2", "min", "grad_norm"):
                    base = {s: r[q] for r in rows if r["method"] == "plain" and r["coords"] == coords
                            and r["policy"] == pol and r["probe"] == probe for s in [r["seed"]]}
                    for m in C.METHODS:
                        vals = {r["seed"]: r[q] for r in rows if r["method"] == m and r["coords"] == coords
                                and r["policy"] == pol and r["probe"] == probe}
                        a = {"coords": coords, "policy": pol, "probe": probe, "quantity": q, "method": m,
                             "comparison": "value", **seedstats(vals)}
                        agg.append(a)
                        if m != "plain":
                            diff = {s: vals[s] - base[s] for s in vals if s in base and np.isfinite(vals[s]) and np.isfinite(base[s])}
                            ratio = {s: vals[s] / base[s] for s in diff if base[s] != 0}
                            agg.append({"coords": coords, "policy": pol, "probe": probe, "quantity": q, "method": m,
                                        "comparison": "difference vs plain", **seedstats(diff)})
                            agg.append({"coords": coords, "policy": pol, "probe": probe, "quantity": q, "method": m,
                                        "comparison": "ratio to plain", **seedstats(ratio)})
    write_csv("D_hessian_summary", agg)
    conv = {"n_problems": len(rows),
            "n_expected": 160,
            "unconverged": [(r["problem"], t) for r in rows for t in TARGETS if not r["%s_converged" % t]],
            "max_iterations": max((r["iterations_start0"] for r in rows), default=None),
            "max_start_rel_diff": {t: max((r.get("%s_start_rel_diff" % t, float("nan")) for r in rows), default=None) for t in TARGETS},
            "n_min_negative_verified": int(sum(r["min_negative_verified"] for r in rows))}
    summary["D_hessian_convergence"] = conv
    summary["D_hessian_summary"] = agg
    figures_spectra(rows, H)
    return rows, H


def figures_spectra(rows, H):
    for q, lab, logy in (("top1", "λ₁ (largest)", True), ("top2", "λ₂", True), ("min", "λ_min (smallest algebraic)", False)):
        fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
        for i, coords in enumerate(C.HESS_COORDS):
            for j, (pol, probe) in enumerate([(p, pr) for p in C.HESS_POLICIES for pr in C.HESS_PROBES]):
                ax = axes[i, j]
                for mi, m in enumerate(C.METHODS):
                    rr = [r for r in rows if r["method"] == m and r["coords"] == coords and r["policy"] == pol and r["probe"] == probe]
                    ok = [r[q] for r in rr if r["%s_converged" % q]]
                    bad = [r["%s_ritz" % q] for r in rr if not r["%s_converged" % q]]
                    ax.scatter([mi] * len(ok), ok, color=COL[m], s=16)
                    ax.scatter([mi] * len(bad), bad, color=COL[m], s=30, marker="x", label="unconverged" if bad else None)
                    if ok:
                        ax.plot([mi - 0.3, mi + 0.3], [np.mean(ok)] * 2, color="k", lw=1.2)
                ax.set_xticks(range(4), [LAB[m] for m in C.METHODS], rotation=25, fontsize=6.5)
                if logy:
                    ax.set_yscale("log")
                else:
                    ax.axhline(0, color="k", lw=0.5)
                ax.set_title("%s H, %s, %s" % (coords, POL[pol], SPL[probe]), fontsize=7)
                ax.set_ylabel(lab)
                ax.grid(alpha=0.3, axis="y")
        fig.suptitle("Hessian restricted to conv + fc weights at the epoch-30 solutions (dots = seeds, bar = mean; × = unconverged Ritz value)")
        save(fig, "D1_hessian_%s" % q)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, t in zip(axes, TARGETS):
        for pid, h in H.items():
            m = parse_pid(pid)[0]
            for sk, st in h["starts"].items():
                it = [x["iter"] for x in st["history"]]
                rr = [x["rel_resid_est"][t] for x in st["history"]]
                ax.plot(it, np.maximum(rr, 1e-18), color=COL[m], lw=0.5, alpha=0.35, ls="-" if sk == "0" else ":")
        ax.axhline(1e-3, color="k", ls="--", lw=0.8)
        ax.set_yscale("log"); ax.set_xlabel("Lanczos iteration"); ax.set_ylabel("residual estimate / scale")
        ax.set_title(t)
        ax.grid(alpha=0.3)
    fig.suptitle("Convergence histories of all eigenproblems (solid = start 0, dotted = start 1; dashed = tolerance)")
    save(fig, "D2_lanczos_convergence")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for i, t in enumerate(("top1", "min")):
        for j, coords in enumerate(C.HESS_COORDS):
            ax = axes[i, j]
            for r in rows:
                if r["coords"] != coords:
                    continue
                ts = sorted(float(k.split("_Cfd_t")[1]) for k in r if k.startswith("%s_Cfd_t" % t))
                y = [r["%s_Cfd_t%s" % (t, C.fnum(x))] / r["%s_quad_r1_f64" % t] for x in ts]
                ax.plot(ts, y, color=COL[r["method"]], lw=0.6, alpha=0.5, marker="." if r["policy"] == C.SAVED else None)
            ax.axhline(1, color="k", lw=0.7)
            ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=0.1)
            ax.set_xlabel("t (r = t)"); ax.set_ylabel("2S(t)/t² ÷ δᵀHδ")
            ax.set_title("%s eigendirection, %s coordinates" % (t, coords))
            ax.grid(alpha=0.3)
    fig.suptitle("Float64 finite differences along the extreme eigendirections vs the HVP quadratic form (1 = agreement)")
    save(fig, "D3_eigendirection_finite_difference")


def quadforms(D, summary):
    out, quad = [], {}
    for tid, rows in D.extra.items():
        for k, r in rows.items():
            if k.startswith("quad|"):
                out.append({"method": r["m"], "seed": r["s"], "probe": r["probe"], "policy": r["policy"], "direction": r["k"],
                            "dHd": r["dHd"], "g_dot_d": r["g_dot_d"], "L0": r["L0"], "grad_norm": r["grad_norm"]})
    write_csv("D_quadratic_forms", out)
    for m in C.METHODS:
        for s in C.SEEDS:
            for probe in C.HESS_PROBES:
                for pol in C.HESS_POLICIES:
                    v = [r["dHd"] for r in out if r["method"] == m and r["seed"] == s and r["probe"] == probe and r["policy"] == pol]
                    if len(v) == C.N_DIRECTIONS:
                        quad[(m, s, probe, pol)] = float(np.mean(v))
    summary["D_quadratic_forms_n"] = len(out)
    return quad, out


def quad_vs_fd(per_seed_rows, quad, summary):
    rows = []
    for (m, s, probe, pol), q in quad.items():
        for amp in C.AMPLITUDES:
            r = [x for x in per_seed_rows if x["method"] == m and x["seed"] == s and x["split"] == probe
                 and x["policy"] == pol and x["amplitude"] == amp]
            if r:
                rows.append({"method": m, "seed": s, "probe": probe, "policy": pol, "amplitude": amp,
                             "mean_dHd": q, "C_mean_finite": r[0]["C_mean"], "ratio_C_over_dHd": r[0]["C_mean"] / q})
    write_csv("D_quadratic_vs_finite_difference", rows)
    summary["D_quad_vs_fd"] = rows
    return rows


def hchecks(D, summary):
    rows = []
    for tid, rr in D.extra.items():
        for k, r in rr.items():
            if k.startswith("hchecks|"):
                row = {kk: v for kk, v in r.items() if not isinstance(v, dict)}
                for lab in ("all_masked", "fc_only"):
                    for h, e in r.get("fd_gradient_vs_hvp_rel_err_%s" % lab, {}).items():
                        row["fd_err_%s_h%s" % (lab, h)] = e
                rows.append(row)
    write_csv("D_hvp_checks", rows)
    summary["D_hvp_checks"] = rows
    return rows


def layer_energy(D, summary):
    """Distribution of r^2 (relative displacement energy) of the r-normalised extreme directions across layers."""
    from continuation_core.models import build_model
    from landscape_study.directions import in_mask
    model = build_model("resnet20_bn_cifar", 10)
    names = [n for n, p in model.named_parameters() if in_mask(n, p)]
    shapes = {n: p.shape for n, p in model.named_parameters()}
    groups = {"conv1": ["conv1.weight"], "stage 1": ["blocks.%d.conv%d.weight" % (b, c) for b in range(3) for c in (1, 2)],
              "stage 2": ["blocks.%d.conv%d.weight" % (b, c) for b in range(3, 6) for c in (1, 2)],
              "stage 3": ["blocks.%d.conv%d.weight" % (b, c) for b in range(6, 9) for c in (1, 2)], "fc": ["fc.weight"]}
    rows = []
    H = D.hessian()
    cks = {}
    for pid in sorted(H):
        m, s, probe, pol, coords = parse_pid(pid)
        vecs = D.vectors(pid)
        if vecs is None:
            continue
        if (m, s) not in cks:
            cks[(m, s)] = torch.load(C.v2_run_dir(m, s) / "checkpoints" / C.FINAL, map_location="cpu", weights_only=False)["model_state"]
        st = cks[(m, s)]
        for vk in TARGETS:
            y = vecs[vk].double()
            off, rel_e, abs_e = 0, {}, {}
            for n in names:
                k = int(np.prod(shapes[n]))
                w = st[n].double().reshape(shapes[n][0], -1)
                bn = w.norm(dim=1)
                yy = y[off:off + k].reshape(shapes[n][0], -1)
                off += k
                delta = yy * bn[:, None] if coords == "relative" else yy
                rel_e[n] = float(((delta.norm(dim=1) ** 2) / torch.where(bn > 0, bn ** 2, torch.ones_like(bn))).sum())
                abs_e[n] = float((delta ** 2).sum())
            tr, ta = sum(rel_e.values()), sum(abs_e.values())
            rec = {"problem": pid, "method": m, "seed": s, "probe": probe, "policy": pol, "coords": coords, "vector": vk}
            for g, ns in groups.items():
                rec["r2_frac_" + g] = sum(rel_e[n] for n in ns) / tr
                rec["l2_frac_" + g] = sum(abs_e[n] for n in ns) / ta
            rows.append(rec)
    write_csv("D_eigendirection_layer_energy", rows)
    summary["D_layer_energy_groups"] = list(groups)
    if rows:
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True, sharey=True)
        for i, coords in enumerate(C.HESS_COORDS):
            for j, vk in enumerate(TARGETS):
                ax = axes[i, j]
                for mi, m in enumerate(C.METHODS):
                    rr = [r for r in rows if r["method"] == m and r["coords"] == coords and r["vector"] == vk
                          and r["policy"] == C.CFROZEN and r["probe"] == "train_probe"]
                    if not rr:
                        continue
                    bottom = 0.0
                    for gi, g in enumerate(groups):
                        v = np.mean([r["r2_frac_" + g] for r in rr])
                        ax.bar(mi, v, bottom=bottom, color=plt.cm.tab10(gi), label=g if mi == 0 else None)
                        bottom += v
                ax.set_xticks(range(4), [LAB[m] for m in C.METHODS], rotation=20, fontsize=7)
                ax.set_title("%s H, %s eigendirection" % (coords, vk))
                ax.set_ylabel("share of r² (mean over seeds)")
        axes[0, 0].legend(frameon=False, fontsize=7)
        fig.suptitle("Where the r-normalised extreme eigendirections put their relative displacement (centre-frozen BN, train probe)")
        save(fig, "D4_eigendirection_layer_energy")
    return rows


def cuts(D, summary, Sbar):
    rows = []
    for m in C.METHODS:
        tgt = C.target(m)
        for s in C.SEEDS:
            for probe in C.HESS_PROBES:
                for pol in C.HESS_POLICIES:
                    for coords in C.HESS_COORDS:
                        pid = C.hess_problem(m, s, probe, pol, coords)
                        vks = TARGETS if coords == "relative" else ("top1", "min")
                        evals = [pol] + ([C.POINTWISE] if (probe == "train_probe" and pol == C.CFROZEN and coords == "relative") else [])
                        for vk in vks:
                            for ep in evals:
                                if ep == C.POINTWISE and vk == "top2":
                                    continue
                                for t in C.CUT_T:
                                    r = D.rows.get(C.pkey(m, s, C.FINAL, tgt, ep, "probes", C.spec_h1(pid, vk, t)))
                                    if r is None:
                                        continue
                                    rows.append({"problem": pid, "method": m, "seed": s, "source_probe": probe, "source_policy": pol,
                                                 "coords": coords, "vector": vk, "eval_policy": ep, "t": t,
                                                 "train_probe_ce": r["splits"]["train_probe"]["ce"],
                                                 "test_probe_ce": r["splits"]["test_probe"]["ce"],
                                                 "train_probe_acc": r["splits"]["train_probe"]["acc"],
                                                 "test_probe_acc": r["splits"]["test_probe"]["acc"]})
    write_csv("D_hessian_cuts", rows)
    summary["D_cuts_n"] = len(rows)
    if not rows:
        return rows
    for pol in C.HESS_POLICIES:
        for coords in C.HESS_COORDS:
            vks = TARGETS if coords == "relative" else ("top1", "min")
            fig, axes = plt.subplots(len(vks), 4, figsize=(14, 2.9 * len(vks)), constrained_layout=True, sharex=True, sharey="row")
            axes = np.atleast_2d(axes)
            for i, vk in enumerate(vks):
                for j, m in enumerate(C.METHODS):
                    ax = axes[i, j]
                    for s in C.SEEDS:
                        for sp, ls in (("train_probe", "-"), ("test_probe", "--")):
                            rr = sorted([r for r in rows if r["method"] == m and r["seed"] == s and r["source_probe"] == "train_probe"
                                         and r["source_policy"] == pol and r["coords"] == coords and r["vector"] == vk
                                         and r["eval_policy"] == pol], key=lambda r: r["t"])
                            if rr:
                                ax.plot([r["t"] for r in rr], [r[sp + "_ce"] for r in rr], ls, color=COL[m], lw=0.8,
                                        alpha=0.9 if s == 0 else 0.45, label=("%s (seed lines)" % SPL[sp]) if (s == 0 and j == 0 and i == 0) else None)
                    ax.set_yscale("log")
                    ax.grid(alpha=0.3)
                    if i == 0:
                        ax.set_title(LAB[m])
                    if j == 0:
                        ax.set_ylabel("CE along %s direction" % vk)
                    if i == len(vks) - 1:
                        ax.set_xlabel("t (RMS relative block displacement r)")
            axes[0, 0].legend(frameon=False, fontsize=6.5)
            fig.suptitle("Cuts along %s-coordinate Hessian eigendirections (train-probe Hessian, %s; evaluated under the same BN policy)" % (coords, POL[pol]))
            save(fig, "D5_cuts_%s_%s" % (coords, pol))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True, sharex=True, sharey="row")
    for i, vk in enumerate(("top1", "min")):
        for j, m in enumerate(C.METHODS):
            ax = axes[i, j]
            for ep, ls in ((C.CFROZEN, "-"), (C.POINTWISE, "--")):
                for s in C.SEEDS:
                    rr = sorted([r for r in rows if r["method"] == m and r["seed"] == s and r["source_probe"] == "train_probe"
                                 and r["source_policy"] == C.CFROZEN and r["coords"] == "relative" and r["vector"] == vk
                                 and r["eval_policy"] == ep], key=lambda r: r["t"])
                    if rr:
                        ax.plot([r["t"] for r in rr], [r["test_probe_ce"] for r in rr], ls, color=COL[m], lw=0.8,
                                alpha=0.45 if s else 0.9, label=POL[ep] if (s == 0 and j == 0 and i == 0) else None)
            ax.set_yscale("log"); ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(LAB[m])
            if j == 0:
                ax.set_ylabel("test-probe CE along relative %s" % vk)
            if i == 1:
                ax.set_xlabel("t (r)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Same eigendirections (centre-frozen train-probe Hessian): frozen statistics vs recalibration at every point")
    save(fig, "D6_cuts_frozen_vs_pointwise")
    return rows

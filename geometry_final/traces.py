"""Trace analysis: per-draw tables, per-checkpoint estimates, paired comparisons, rank agreement.

    py -m geometry_final.traces [--root studies/geometry_final/raw/main/gf]

Monte Carlo uncertainty (SEM over Rademacher draws within one checkpoint) and
seed-to-seed variation (SD over the five training seeds) are kept in separate
columns.  Paired method-minus-Plain differences use the shared draws: the SEM is
that of the per-draw differences.  The control-variate estimate is secondary.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from landscape_v3 import common as V3

from . import common as C
from .data import NEW, blocks, trace_draws
from .trace import summary

TAB = C.STUDY / "tables"
KINDS = C.TRACE_KINDS


def _weights(norm2, sizes):
    return {"ordinary": np.ones_like(norm2), "relative": norm2, "covariance": norm2 / sizes}


def _vectors(pid):
    f = glob.glob(str(C.V3_RAW / "*" / "v3" / "hessian" / (pid + "_vectors.pt")))
    return torch.load(f[0], map_location="cpu", weights_only=True)


def control_variate(df, qs, root):
    """Per-draw CV estimates using the existing top1/top2 ordinary centre-frozen eigenpairs."""
    prob = pd.read_csv(C.V3_TABLES / "D_hessian_problems.csv").set_index("problem")
    out = []
    for oid, g in df.groupby("objective"):
        m, s, probe = g.m.iloc[0], int(g.s.iloc[0]), g.probe.iloc[0]
        pid = V3.hess_problem(m, s, probe, C.CFROZEN, "ordinary")
        vec = _vectors(pid)
        norm2, sizes = blocks(m, s, root)
        bid = np.repeat(np.arange(len(sizes)), sizes)
        W = {k: w[bid] for k, w in _weights(norm2, sizes).items()}
        pairs = [(float(prob.loc[pid, vk]), vec[vk].double().numpy()) for vk in ("top1", "top2")]
        pairs = [(lam, u / np.linalg.norm(u)) for lam, u in pairs]
        for _, r in g.iterrows():
            z = C.rademacher(s, int(r.draw), len(bid)).astype(np.float64)
            rec = {"objective": oid, "draw": int(r.draw)}
            for k in KINDS:
                corr = 0.0
                for lam, u in pairs:
                    corr += lam * ((z * W[k] * u).sum() * (u @ z) - (W[k] * u * u).sum())
                rec["trace_%s_cv" % k] = r["trace_" + k] - corr
            out.append(rec)
    return pd.DataFrame(out)


def per_checkpoint(df, cv):
    prob = pd.read_csv(C.V3_TABLES / "D_hessian_problems.csv").set_index("problem")
    quad = pd.read_csv(C.V3_TABLES / "D_quadratic_forms.csv")
    quad = quad[quad.policy == "centre_frozen"]
    d = df.merge(cv, on=["objective", "draw"])
    rows = []
    for oid, g in d.groupby("objective"):
        m, s, probe = g.m.iloc[0], int(g.s.iloc[0]), g.probe.iloc[0]
        po = prob.loc[V3.hess_problem(m, s, probe, C.CFROZEN, "ordinary")]
        pr = prob.loc[V3.hess_problem(m, s, probe, C.CFROZEN, "relative")]
        qf = quad[(quad.method == m) & (quad.seed == s) & (quad.probe == probe)].dHd.to_numpy()
        r = {"objective": oid, "method": m, "seed": s, "probe": probe, "bn_policy": "centre_frozen",
             "parameter_subset": "conv+fc weights (268,336; 698 blocks)", "n_draws": len(g),
             "lambda_top1_ordinary": po.top1, "lambda_top2_ordinary": po.top2, "lambda_min_ordinary": po["min"],
             "lambda_top1_relative": pr.top1, "lambda_min_relative": pr["min"], "G": int(pr.n_nonzero_blocks), "P": int(pr.n_params),
             "loss": po.loss, "grad_norm": po.grad_norm}
        for k in KINDS:
            sm = summary(g["trace_" + k])
            smc = summary(g["trace_%s_cv" % k])
            r.update({"trace_%s" % k: sm["mean"], "trace_%s_mc_sd" % k: sm["sd"], "trace_%s_mc_sem" % k: sm["sem"],
                      "trace_%s_mc_sem_rel" % k: sm["sem_rel"], "trace_%s_meets_5pct" % k: sm["sem_rel"] <= C.TRACE_SEM_REL,
                      "trace_%s_cv" % k: smc["mean"], "trace_%s_cv_mc_sem" % k: smc["sem"]})
        r["mean_eigenvalue_ordinary"] = r["trace_ordinary"] / r["P"]
        r["lambda_top1_over_mean_eigenvalue"] = r["lambda_top1_ordinary"] / r["mean_eigenvalue_ordinary"]
        r["top2_share_of_trace_signed"] = (po.top1 + po.top2) / r["trace_ordinary"]
        r["r_units_extreme_top1"] = r["G"] * pr.top1
        r["r_units_isotropic_relative_mean"] = r["G"] * r["trace_relative"] / r["P"]
        r["quadform_20dir_mean"] = qf.mean()
        r["quadform_20dir_sem"] = qf.std(ddof=1) / np.sqrt(len(qf))
        r["trHC_minus_quadform_z"] = (r["trace_covariance"] - qf.mean()) / np.hypot(r["trace_covariance_mc_sem"], r["quadform_20dir_sem"])
        r["covariance_trace_over_extreme_r_units"] = r["trace_covariance"] / r["r_units_extreme_top1"]
        rows.append(r)
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(C.METHODS)}
    return out.sort_values(["probe", "seed", "method"], key=lambda c: c.map(order) if c.name == "method" else c).reset_index(drop=True)


def paired(df, pc):
    rows = []
    for (s, probe), g in df.groupby(["s", "probe"]):
        base = g[g.m == "plain"].set_index("draw")
        for m in C.METHODS[1:]:
            gm = g[g.m == m].set_index("draw")
            common = sorted(set(base.index) & set(gm.index))
            for k in KINDS:
                diff = gm.loc[common, "trace_" + k].to_numpy() - base.loc[common, "trace_" + k].to_numpy()
                sm = summary(diff)
                a, b = gm.loc[common, "trace_" + k].to_numpy(), base.loc[common, "trace_" + k].to_numpy()
                ratio = a.mean() / b.mean()
                # delta-method SEM of the ratio of paired means
                cov = np.cov(a, b, ddof=1) / len(common)
                ratio_sem = abs(ratio) * np.sqrt(cov[0, 0] / a.mean() ** 2 + cov[1, 1] / b.mean() ** 2 - 2 * cov[0, 1] / (a.mean() * b.mean()))
                rows.append({"seed": s, "probe": probe, "method": m, "quantity": "trace_" + k, "n_shared_draws": len(common),
                             "diff_vs_plain": sm["mean"], "diff_mc_sem": sm["sem"], "diff_z": sm["mean"] / sm["sem"],
                             "diff_resolved_2sem": abs(sm["mean"]) > 2 * sm["sem"],
                             "ratio_to_plain": ratio, "ratio_mc_sem": ratio_sem,
                             "unpaired_sem_would_be": np.hypot(a.std(ddof=1), b.std(ddof=1)) / np.sqrt(len(common))})
        pcs = pc[(pc.seed == s) & (pc.probe == probe)].set_index("method")
        for m in C.METHODS[1:]:
            for q in ("lambda_top1_ordinary", "lambda_top1_relative", "quadform_20dir_mean"):
                rows.append({"seed": s, "probe": probe, "method": m, "quantity": q, "diff_vs_plain": pcs.loc[m, q] - pcs.loc["plain", q],
                             "ratio_to_plain": pcs.loc[m, q] / pcs.loc["plain", q],
                             "diff_mc_sem": pcs.loc[m, "quadform_20dir_sem"] if q == "quadform_20dir_mean" else np.nan})
    return pd.DataFrame(rows)


def seed_summary(pc, pr):
    rows = []
    qty = ["lambda_top1_ordinary", "lambda_top1_relative", "trace_ordinary", "trace_relative", "trace_covariance", "quadform_20dir_mean",
           "lambda_min_ordinary"]
    for probe in C.PROBES:
        for m in C.METHODS:
            g = pc[(pc.probe == probe) & (pc.method == m)]
            for q in qty:
                v = g[q].to_numpy()
                r = {"probe": probe, "method": m, "quantity": q, "n_seeds": len(v), "mean": v.mean(), "sd_over_seeds": v.std(ddof=1),
                     "min": v.min(), "max": v.max()}
                if q.startswith("trace_"):
                    r["mean_mc_sem"] = g[q + "_mc_sem"].mean()
                    r["max_mc_sem_rel"] = g[q + "_mc_sem_rel"].max()
                    r["draws_min"], r["draws_max"] = int(g.n_draws.min()), int(g.n_draws.max())
                if m != "plain":
                    p = pr[(pr.probe == probe) & (pr.method == m) & (pr.quantity == q)]
                    if len(p):
                        r.update({"ratio_to_plain_mean": p.ratio_to_plain.mean(), "ratio_to_plain_sd": p.ratio_to_plain.std(ddof=1),
                                  "n_seeds_below_plain": int((p.diff_vs_plain < 0).sum())})
                        if q.startswith("trace_"):
                            r["n_seeds_below_plain_resolved_2sem"] = int(((p.diff_vs_plain < 0) & p.diff_resolved_2sem.astype(bool)).sum())
                            r["n_seeds_above_plain_resolved_2sem"] = int(((p.diff_vs_plain > 0) & p.diff_resolved_2sem.astype(bool)).sum())
                            r["n_seeds_unresolved"] = int((~p.diff_resolved_2sem.astype(bool)).sum())
                rows.append(r)
    return pd.DataFrame(rows)


def rank_agreement(pc, pr):
    """Within each seed and probe: does 'smaller than Plain' on lambda_max coincide with 'smaller than Plain' on each trace?"""
    rows = []
    for (s, probe), g in pc.groupby(["seed", "probe"]):
        for q in ("lambda_top1_ordinary", "lambda_top1_relative", "trace_ordinary", "trace_relative", "trace_covariance", "quadform_20dir_mean"):
            order = g.sort_values(q).method.map(dict(zip(C.METHODS, ["Plain", "R", "G", "RG"]))).tolist()
            rows.append({"seed": s, "probe": probe, "quantity": q, "ascending_order": " < ".join(order)})
    rk = pd.DataFrame(rows)
    agree = []
    for (s, probe, m), g in pr[pr.quantity.isin(["lambda_top1_ordinary", "trace_ordinary", "lambda_top1_relative", "trace_relative", "trace_covariance"])].groupby(["seed", "probe", "method"]):
        gg = g.set_index("quantity")
        rec = {"seed": s, "probe": probe, "method": m}
        for q in gg.index:
            rec[q + "_below_plain"] = bool(gg.loc[q, "diff_vs_plain"] < 0)
            if q.startswith("trace_"):
                rec[q + "_resolved_2sem"] = bool(gg.loc[q, "diff_resolved_2sem"])
        rec["ordinary_lambda_and_trace_agree"] = rec["lambda_top1_ordinary_below_plain"] == rec["trace_ordinary_below_plain"]
        rec["relative_lambda_and_trace_agree"] = rec["lambda_top1_relative_below_plain"] == rec["trace_relative_below_plain"]
        agree.append(rec)
    return rk, pd.DataFrame(agree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(NEW))
    a = ap.parse_args()
    root = Path(a.root)
    TAB.mkdir(parents=True, exist_ok=True)
    df, qs = trace_draws(root)
    keep = ["objective", "m", "s", "probe", "draw", "rademacher_seed", "z_sha256", "trace_ordinary", "trace_relative",
            "trace_covariance", "zHz_direct", "seconds"]
    df[keep].rename(columns={"m": "method", "s": "seed"}).to_csv(TAB / "T_trace_draws.csv", index=False)
    arrays = {}
    for oid, g in df.groupby("objective"):
        arrays[oid] = np.stack([qs[(oid, d)] for d in g.draw])
        arrays[oid + "__draws"] = g.draw.to_numpy()
    np.savez_compressed(TAB / "T_trace_q_blocks.npz", **arrays)
    cv = control_variate(df, qs, root)
    cv.to_csv(TAB / "T_trace_draws_control_variate.csv", index=False)
    pc = per_checkpoint(df, cv)
    pc.to_csv(TAB / "T_trace_per_checkpoint.csv", index=False)
    pr = paired(df, pc)
    pr.to_csv(TAB / "T_trace_paired_vs_plain.csv", index=False)
    ss = seed_summary(pc, pr)
    ss.to_csv(TAB / "T_trace_seed_summary.csv", index=False)
    rk, ag = rank_agreement(pc, pr)
    rk.to_csv(TAB / "T_rank_orders.csv", index=False)
    ag.to_csv(TAB / "T_lambda_trace_agreement.csv", index=False)
    print("draws", len(df), "objectives", df.objective.nunique())


if __name__ == "__main__":
    main()

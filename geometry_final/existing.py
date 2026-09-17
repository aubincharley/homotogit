"""Analyses of already-computed landscape_v3 measurements (no GPU, no new evaluation).

    py -m geometry_final.existing

Writes to ``studies/geometry_final/tables/``:

* ``X1_bn_policy_finite_per_seed.csv`` / ``X1_bn_policy_finite_summary.csv``
  identical random perturbations (block A) under centre-frozen, pointwise and saved BN;
* ``X2_bn_policy_eigencuts.csv``  the same eigendirection cuts under centre-frozen and pointwise BN;
* ``X3_direction_curvature_per_checkpoint.csv``  extreme eigenvalues vs random quadratic forms,
  one frozen objective at a time, in ordinary units and in r-units;
* ``X4_amplitude_random.csv`` / ``X4_amplitude_eigen.csv``  local HVP quadratic forms vs finite
  differences along the same directions;
* ``X5_eigvec_radial_fraction.csv``  share of each stored extreme eigenvector that rescales whole
  filters/rows (the component frozen BN does not normalise away).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from landscape_v3 import common as V3

from . import common as C

OUT = C.STUDY / "tables"
T = C.V3_TABLES


def _ord(df, col="method"):
    order = {m: i for i, m in enumerate(C.METHODS)}
    return df.sort_values(by=[col], key=lambda s: s.map(order), kind="stable")


def sign_summary(df, keys, value):
    g = df.groupby(keys)[value]
    return pd.DataFrame({"n_seeds": g.size(), "mean": g.mean(), "sd": g.std(ddof=1), "min": g.min(), "max": g.max(),
                         "n_negative": g.apply(lambda v: int((v < 0).sum())),
                         "n_positive": g.apply(lambda v: int((v > 0).sum()))}).reset_index()


# ---- 1. BN policy on identical finite perturbations -------------------------------------

def bn_policy_finite():
    a = pd.read_csv(T / "A_sensitivity_per_seed.csv")
    w = a.pivot_table(index=["method", "seed", "split", "amplitude"], columns="policy", values="S_mean").reset_index()
    w.columns.name = None
    w = w.rename(columns={"centre_frozen": "S_centre_frozen", "recalibrated": "S_pointwise", "saved": "S_saved"})
    w["ratio_pointwise_over_centre_frozen"] = w.S_pointwise / w.S_centre_frozen
    w["ratio_saved_over_centre_frozen"] = w.S_saved / w.S_centre_frozen
    plain = w[w.method == "plain"].set_index(["seed", "split", "amplitude"])
    for pol in ("centre_frozen", "pointwise", "saved"):
        w["delta_vs_plain_" + pol] = w["S_" + pol].values - plain.loc[
            list(zip(w.seed, w.split, w.amplitude)), "S_" + pol].values
        w["relchange_vs_plain_" + pol] = w["delta_vs_plain_" + pol].values / plain.loc[
            list(zip(w.seed, w.split, w.amplitude)), "S_" + pol].values
    w["delta_sign_differs_cf_vs_pw"] = np.sign(w.delta_vs_plain_centre_frozen) != np.sign(w.delta_vs_plain_pointwise)
    w = _ord(w)
    w.to_csv(OUT / "X1_bn_policy_finite_per_seed.csv", index=False)
    rows = []
    for (m, sp, amp), g in w.groupby(["method", "split", "amplitude"]):
        r = {"method": m, "split": sp, "amplitude": amp, "n_seeds": len(g),
             "ratio_pw_over_cf_median": g.ratio_pointwise_over_centre_frozen.median(),
             "ratio_pw_over_cf_min": g.ratio_pointwise_over_centre_frozen.min(),
             "ratio_pw_over_cf_max": g.ratio_pointwise_over_centre_frozen.max()}
        if m != "plain":
            for pol in ("centre_frozen", "pointwise", "saved"):
                d = g["delta_vs_plain_" + pol]
                r.update({"delta_%s_mean" % pol: d.mean(), "delta_%s_sd" % pol: d.std(ddof=1),
                          "relchange_%s_mean" % pol: g["relchange_vs_plain_" + pol].mean(),
                          "n_below_plain_%s" % pol: int((d < 0).sum())})
            r["n_seeds_sign_differs_cf_vs_pw"] = int(g.delta_sign_differs_cf_vs_pw.sum())
        rows.append(r)
    s = _ord(pd.DataFrame(rows))
    s.to_csv(OUT / "X1_bn_policy_finite_summary.csv", index=False)
    return w, s


def bn_policy_eigencuts():
    c = pd.read_csv(T / "D_hessian_cuts.csv")
    c = c[(c.source_policy == "centre_frozen") & (c.coords == "relative") & (c.source_probe == "train_probe")
          & c.vector.isin(["top1", "min"])]
    w = c.pivot_table(index=["method", "seed", "vector", "t"], columns="eval_policy",
                      values=["train_probe_ce", "test_probe_ce"]).reset_index()
    w.columns = ["_".join([x for x in col if x]) for col in w.columns]
    for sp in ("train_probe", "test_probe"):
        for pol in ("centre_frozen", "recalibrated"):
            centre = w[w.t == 0].set_index(["method", "seed", "vector"])["%s_ce_%s" % (sp, pol)]
            w["%s_rise_%s" % (sp, pol)] = w["%s_ce_%s" % (sp, pol)].values - centre.loc[
                list(zip(w.method, w.seed, w.vector))].values
    w = w.rename(columns=lambda x: x.replace("recalibrated", "pointwise"))
    w = _ord(w)
    w.to_csv(OUT / "X2_bn_policy_eigencuts.csv", index=False)
    return w


# ---- 2. direction: extreme vs random curvature, same frozen objective ---------------------

def direction_curvature():
    p = pd.read_csv(T / "D_hessian_problems.csv")
    q = pd.read_csv(T / "D_quadratic_forms.csv")
    rows = []
    for (m, s, probe, pol), g in q.groupby(["method", "seed", "probe", "policy"]):
        ordp = p[(p.method == m) & (p.seed == s) & (p.probe == probe) & (p.policy == pol) & (p.coords == "ordinary")].iloc[0]
        relp = p[(p.method == m) & (p.seed == s) & (p.probe == probe) & (p.policy == pol) & (p.coords == "relative")].iloc[0]
        G, P = int(relp.n_nonzero_blocks), int(relp.n_params)
        d = g.dHd.values
        rows.append({
            "method": m, "seed": s, "probe": probe, "policy": pol, "G": G, "P": P,
            "lambda_top1_ordinary": ordp.top1, "lambda_top2_ordinary": ordp.top2, "lambda_min_ordinary": ordp["min"],
            "lambda_top1_relative": relp.top1, "lambda_top2_relative": relp.top2, "lambda_min_relative": relp["min"],
            "grad_norm": ordp.grad_norm, "loss": ordp.loss,
            # r-units: curvature per unit r^2 (r = RMS relative block displacement)
            "r_units_extreme_top1": G * relp.top1, "r_units_extreme_min": G * relp["min"],
            "r_units_random_mean_dHd": d.mean(), "r_units_random_sd_dHd": d.std(ddof=1),
            "r_units_random_sem_dHd": d.std(ddof=1) / np.sqrt(len(d)),
            "r_units_random_min_dHd": d.min(), "r_units_random_max_dHd": d.max(),
            "n_random_directions": len(d), "n_random_dHd_negative": int((d < 0).sum()),
            "random_mean_over_extreme_top1": d.mean() / (G * relp.top1),
            "top1_quad_r1_f64_measured": ordp.top1_quad_r1_f64, "min_quad_r1_f64_measured": ordp.min_quad_r1_f64,
        })
    df = _ord(pd.DataFrame(rows))
    df.to_csv(OUT / "X3_direction_curvature_per_checkpoint.csv", index=False)
    return df


# ---- 3. amplitude: local quadratic vs finite differences --------------------------------

def amplitude():
    fd = pd.read_csv(T / "D_quadratic_vs_finite_difference.csv")
    s = fd.groupby(["policy", "probe", "method", "amplitude"]).ratio_C_over_dHd.agg(
        n_seeds="size", median="median", min="min", max="max").reset_index()
    s = _ord(s)
    s.to_csv(OUT / "X4_amplitude_random.csv", index=False)
    p = pd.read_csv(T / "D_hessian_problems.csv")
    rows = []
    for _, r in p.iterrows():
        for vk in ("top1", "min"):
            for t in V3.FD_T:
                col = "%s_Cfd_t%s" % (vk, V3.fnum(t))
                rows.append({"problem": r.problem, "method": r.method, "seed": r.seed, "probe": r.probe,
                             "policy": r.policy, "coords": r.coords, "vector": vk, "t": t,
                             "C_fd": r[col], "quad_r1_f64": r["%s_quad_r1_f64" % vk],
                             "ratio_C_fd_over_quad": r[col] / r["%s_quad_r1_f64" % vk],
                             "sign_agrees": bool(np.sign(r[col]) == np.sign(r["%s_quad_r1_f64" % vk]))})
    e = _ord(pd.DataFrame(rows))
    e.to_csv(OUT / "X4_amplitude_eigen.csv", index=False)
    return s, e


# ---- 4. radial share of stored eigenvectors ---------------------------------------------

def _centre(m, s):
    ck = torch.load(C.V3_ROOT / "studies" / "landscape_v2" / "raw" / V3.V2_ACCOUNT_OF_SEED[s] / "v2" / "runs"
                    / V3.rname(m, s) / "checkpoints" / C.FINAL, map_location="cpu", weights_only=False)
    return ck["model_state"]


def radial_fraction():
    from continuation_core.models import build_model
    model = build_model("resnet20_bn_cifar", 10)
    names = [n for n, p in model.named_parameters() if n.endswith("weight") and p.dim() in (2, 4)]
    vec_files = {Path(f).name.replace("_vectors.pt", ""): f for f in glob.glob(str(C.V3_RAW / "*" / "v3" / "hessian" / "*_vectors.pt"))}
    rows = []
    for m in C.METHODS:
        for s in C.SEEDS:
            st = _centre(m, s)
            blocks = [st[n].to(torch.float64).reshape(st[n].shape[0], -1) for n in names]
            theta = torch.cat([b.reshape(-1) for b in blocks])
            norms = torch.cat([b.norm(dim=1) for b in blocks])
            sizes = torch.cat([torch.full((b.shape[0],), b.shape[1], dtype=torch.float64) for b in blocks])
            bid = torch.cat([torch.arange(b.shape[0]).repeat_interleave(b.shape[1]) + sum(x.shape[0] for x in blocks[:i])
                             for i, b in enumerate(blocks)])
            that = theta / norms[bid]
            bn_followed = torch.cat([torch.full((b.shape[0],), 0.0 if n == "fc.weight" else 1.0, dtype=torch.float64)
                                     for n, b in zip(names, blocks)])
            G = len(norms)
            baseline_r = float((1.0 / sizes).mean())
            baseline_ord = float(((norms ** 2) / sizes).sum() / (norms ** 2).sum())
            for probe in ("train", "test"):
                for pol in ("saved", "centre_frozen"):
                    for coords in ("ord", "rel"):
                        pid = "%s_s%d_%s_%s_%s" % (C.SHORT[m], s, probe, pol, coords)
                        vecs = torch.load(vec_files[pid], map_location="cpu", weights_only=True)
                        for vk, y in vecs.items():
                            y = y.to(torch.float64)
                            delta = y * norms[bid] if coords == "rel" else y
                            proj = torch.zeros(G, dtype=torch.float64).index_add_(0, bid, delta * that)  # delta_g . theta_hat_g
                            dn2 = torch.zeros(G, dtype=torch.float64).index_add_(0, bid, delta ** 2)
                            q_rad = proj / norms                         # relative-metric radial coordinate
                            q_n2 = dn2 / norms ** 2
                            rows.append({"problem": pid, "method": m, "seed": s, "probe": probe + "_probe", "policy": pol,
                                         "coords": {"ord": "ordinary", "rel": "relative"}[coords], "vector": vk,
                                         "radial_fraction_ordinary_metric": float((proj ** 2).sum() / dn2.sum()),
                                         "radial_fraction_r_metric": float((q_rad ** 2).sum() / q_n2.sum()),
                                         "radial_fraction_bn_followed_r_metric": float(((q_rad ** 2) * bn_followed).sum() / q_n2.sum()),
                                         "fc_share_r_metric": float((q_n2 * (1 - bn_followed)).sum() / q_n2.sum()),
                                         "random_direction_expected_radial_fraction_r_metric": baseline_r,
                                         "random_direction_expected_radial_fraction_ordinary_metric": baseline_ord})
    df = _ord(pd.DataFrame(rows))
    df.to_csv(OUT / "X5_eigvec_radial_fraction.csv", index=False)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bn_policy_finite()
    bn_policy_eigencuts()
    direction_curvature()
    amplitude()
    radial_fraction()
    print("written to", OUT)


if __name__ == "__main__":
    main()

"""Verification run at the start of every job; results in ``checks_job<k>.json``.

1. direction normalisation: every non-zero block of every direction used has
   ``||D_j|| / ||W_j|| = 1``; unmasked tensors and zero-norm blocks get zero;
2. reconstruction: ``centre + 0*D + 0*E`` cast to float32 equals the checkpoint
   parameters bitwise;
3. interpolation endpoints: alpha = 0 and 1 give the exact endpoint parameters;
4. PCA plane: stored coordinates reconstruct the stored residuals;
5. exact bypass: the block-1 reduction model at r = 32 gives bitwise the logits
   of the hook-free model (inference and training mode) and the same
   recalibrated loss;
6. determinism and order independence of recalibrated evaluation;
7. non-mutation of the checkpoint files and of the loaded tensors;
8. calibration indices disjoint from the training probe; test never used.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from continuation_core.analysis.params import to_vector

from .directions import cosine, normalization_report, scale_to
from .evaluator import weights_sha
from .sources import sha_file

CENTERS = [("plain", s, 30) for s in (0, 1, 2)] + [("resolution_max_b1", s, 30) for s in (0, 1, 2)] \
    + [("resolution_max_b1", 0, 6), ("resolution_max_b1", 0, 12), ("plain", 0, 6)]


def run_checks(ev, inp, path, device) -> dict:
    ev.model("plain")
    ev.model("resolution_max_b1")
    names = ev.names
    res = {}
    wname = lambda m, s, e: "%s__seed%d__ep%02d.pt" % (m, s, e)
    file_sha_before = {wname(*c): sha_file(inp.root / "weights" / wname(*c)) for c in CENTERS}
    tensor_sha_before = {k: weights_sha(inp.weights(k)["model_state"], names) for k in file_sha_before}

    norm, recon = {}, {}
    for c in CENTERS:
        k = wname(*c)
        cs = {n: inp.weights(k)["model_state"][n] for n in names}
        for pair in range(5):
            for ax in ("D", "E"):
                d, _ = scale_to(cs, inp.draws[pair][ax], names)
                norm["%s|pair%d|%s" % (k, pair, ax)] = normalization_report(cs, d, names)
        dD, _ = scale_to(cs, inp.draws[0]["D"], names)
        dE, _ = scale_to(cs, inp.draws[0]["E"], names)
        norm["%s|pair0|cos(D,E)" % k] = cosine(dD, dE, names)
        rebuilt = {n: (cs[n].double() + 0.0 * dD[n] + 0.0 * dE[n]).float() for n in names}
        recon[k] = all(torch.equal(rebuilt[n], cs[n]) for n in names)
    res["direction_normalization"] = {
        "max_abs_relative_norm_error": max(v["max_abs_relative_norm_error"]
                                          for v in norm.values() if isinstance(v, dict)),
        "all_zero_rules_hold": all(v["unmasked_and_zero_blocks_have_zero_direction"]
                                   for v in norm.values() if isinstance(v, dict)),
        "n_zero_norm_blocks_total": sum(v["n_zero_norm_blocks"] for v in norm.values()
                                        if isinstance(v, dict)),
        "masked_tensors": next(v for v in norm.values() if isinstance(v, dict))["masked_tensors"],
        "cos_D_E_pair0": {k: v for k, v in norm.items() if not isinstance(v, dict)},
        "per_direction": norm}
    res["reconstruction_bitwise"] = recon

    ends = {}
    for s in (0, 1, 2):
        A = {n: inp.weights(wname("plain", s, 30))["model_state"][n] for n in names}
        B = {n: inp.weights(wname("resolution_max_b1", s, 30))["model_state"][n] for n in names}
        a0 = {n: ((1.0 - 0.0) * A[n].double() + 0.0 * B[n].double()).float() for n in names}
        a1 = {n: ((1.0 - 1.0) * A[n].double() + 1.0 * B[n].double()).float() for n in names}
        ends["seed%d" % s] = {"alpha0_equals_A": all(torch.equal(a0[n], A[n]) for n in names),
                              "alpha1_equals_B": all(torch.equal(a1[n], B[n]) for n in names),
                              "endpoint_distance_l2": float((to_vector(A, names) - to_vector(B, names)).norm())}
    res["interpolation_endpoints"] = ends

    pl = inp.plane
    X = torch.stack([to_vector(inp.weights(p["file"])["model_state"], names) for p in pl["points"]])
    C = X - pl["origin"]
    co = torch.stack([C @ pl["d1"], C @ pl["d2"]], 1)
    resid = (C - co[:, :1] * pl["d1"] - co[:, 1:] * pl["d2"]).norm(dim=1)
    res["pca_plane"] = {"coords_match": bool(torch.allclose(co, torch.tensor(pl["coords"], dtype=torch.float64), atol=1e-8)),
                        "residuals_match": bool(torch.allclose(resid, torch.tensor(pl["residual_norm"], dtype=torch.float64), rtol=1e-8)),
                        "d1_d2_orthonormal": [float(pl["d1"].norm()), float(pl["d2"].norm()), float(pl["d1"] @ pl["d2"])]}

    # exact bypass on real weights
    k = wname("resolution_max_b1", 0, 30)
    st = inp.weights(k)["model_state"]
    mr, cr = ev.model("resolution_max_b1")
    mp, _ = ev.model("plain")
    x = ev.pipeline(ev.splits["test_probe"][0][:500])
    with torch.no_grad():
        mr.load_state_dict({a: b.to(device) for a, b in st.items()})
        mp.load_state_dict({a: b.to(device) for a, b in st.items()})
        cr.set_state(ev_state(32))
        mr.eval(); mp.eval()
        eval_eq = torch.equal(mr(x), mp(x))
        mr.load_state_dict({a: b.to(device) for a, b in st.items()})
        mp.load_state_dict({a: b.to(device) for a, b in st.items()})
        mr.train(); mp.train()
        train_eq = torch.equal(mr(x), mp(x))
        cr.set_state(ev_state(16))
        mr.eval()
        active_differs = not torch.equal(mr(x), mp(x))
    params = {n: st[n] for n in names}
    rr = ev.recalibrated("resolution_max_b1", params, {"resolution": 32, "sigma": None})
    rp = ev.recalibrated("plain", params, {"resolution": None, "sigma": None})
    res["bypass_r32"] = {"eval_logits_bitwise": eval_eq, "train_logits_bitwise": train_eq,
                         "r16_differs_from_plain": active_differs,
                         "recalibrated_equal": rr["splits"] == rp["splits"]}

    # determinism and traversal order
    c64 = {n: st[n].to(device, torch.float64) for n in names}
    dD, _ = scale_to(params, inp.draws[0]["D"], names)
    dE, _ = scale_to(params, inp.draws[0]["E"], names)
    pts = [(0.1, -0.1), (-0.2, 0.05), (0.0, 0.25)]

    def at(a, b):
        p = {n: (c64[n] + a * dD[n].to(device) + b * dE[n].to(device)).float() for n in names}
        return ev.recalibrated("resolution_max_b1", p, {"resolution": 32, "sigma": None})["splits"]

    fwd = [at(*p) for p in pts]
    rev = [at(*p) for p in reversed(pts)][::-1]
    again = at(*pts[0])
    res["determinism"] = {"repeat_identical": again == fwd[0],
                          "order_independent": fwd == rev,
                          "max_abs_ce_diff_order": max(abs(f[s]["ce"] - r[s]["ce"])
                                                       for f, r in zip(fwd, rev) for s in f)}

    res["non_mutation"] = {
        "files_unchanged": all(sha_file(inp.root / "weights" / k) == v for k, v in file_sha_before.items()),
        "loaded_tensors_unchanged": all(weights_sha(inp.weights(k)["model_state"], names) == v
                                        for k, v in tensor_sha_before.items())}
    sub = inp.subsets
    res["subsets"] = {"calibration_disjoint_from_train_probe": not (set(sub["calibration_train_idx"].tolist())
                                                                   & set(sub["train_probe_idx"].tolist())),
                      "calibration_source": "training split",
                      "class_counts": {k: np.bincount(
                          (ev.splits["train_probe"][1] if k == "train_probe_idx" else
                           ev.splits["test_probe"][1] if k == "test_probe_idx" else
                           ev.calibration[1]).cpu().numpy(), minlength=10).tolist()
                          for k in ("train_probe_idx", "test_probe_idx", "calibration_train_idx")}}
    flat = {"normalization_ok": res["direction_normalization"]["max_abs_relative_norm_error"] < 1e-9
            and res["direction_normalization"]["all_zero_rules_hold"],
            "reconstruction_ok": all(recon.values()),
            "endpoints_ok": all(v["alpha0_equals_A"] and v["alpha1_equals_B"] for v in ends.values()),
            "pca_ok": res["pca_plane"]["coords_match"] and res["pca_plane"]["residuals_match"],
            "bypass_ok": eval_eq and train_eq and active_differs and res["bypass_r32"]["recalibrated_equal"],
            "determinism_ok": res["determinism"]["repeat_identical"] and res["determinism"]["order_independent"],
            "non_mutation_ok": all(res["non_mutation"].values()),
            "subsets_ok": res["subsets"]["calibration_disjoint_from_train_probe"]}
    res["summary"] = flat
    path.write_text(json.dumps(res, indent=2, default=str))
    print("checks:", flat, flush=True)
    return res


def ev_state(r):
    from continuation_core.controller import InterventionState
    return InterventionState(r, None, "check")

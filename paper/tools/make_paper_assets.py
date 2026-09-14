"""Figures, tables and number macros of the working paper, generated from records.

Nothing here trains or evaluates a network.  Sources, all read-only:

* the audited experiment index ``experiments/index.json`` and the per-run
  ``summary.json`` / ``metrics.json`` files of branch ``benchmark-organized``
  at the pinned commit ``BENCH_COMMIT``, read through ``git show`` (no checkout
  and no benchmark code on the path).  Cells stored on a teammate's branch are
  read at the commit recorded for them in the index;
* the frozen presets of ``continuation_core`` (this branch) for the schedule
  table.

Run from anywhere inside the repository::

    py paper/tools/make_paper_assets.py

Outputs: ``paper/figures/appendix_*.pdf``, ``paper/tables/*.tex`` (generated
ones start with a ``% GENERATED`` comment) and
``paper/provenance/generated_assets.json``, which lists for every output the
experiments, configuration ids, seeds, cell ids and record files it used.
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BENCH_BRANCH = "benchmark-organized"
BENCH_COMMIT = "8d353f208661e89196ef1ab2b68b3bf4bd597d35"
INDEX_PATH = "experiments/index.json"
REF_NAMES = {"591e125aebc345eac926c05497a903cd6d192f5c": "continuation-gaussian-tv_exploration_1",
             "ea186fe77d03a5797520ca57ffa33838999a07d3": "adaptative-schedule",
             BENCH_COMMIT: BENCH_BRANCH}
COMMAND = "py paper/tools/make_paper_assets.py"

FIG = PAPER / "figures"
TAB = PAPER / "tables"
PROV_OUT = PAPER / "provenance" / "generated_assets.json"

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.5, "axes.titlesize": 8,
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 6.8,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "figure.facecolor": "white"})

# One palette for the paper: plain black, the three retained procedures in
# fixed colours, everything else grey.
COLOR = {"plain": "#000000", "resolution_max_b1": "#0072B2",
         "gaussian_postrelu": "#E69F00", "resolution_max_b1_gaussian_conv": "#7B3294"}
GREY = "#8c8c8c"
PAPER_NAME = {"plain": "Plain", "resolution_max_b1": "Resolution",
              "gaussian_postrelu": "Gaussian", "resolution_max_b1_gaussian_conv": "Combined"}
UNIFIED_ID = {"plain": "plain", "resolution_max_b1": "shrink_b1",
              "gaussian_postrelu": "blur_relu", "resolution_max_b1_gaussian_conv": "shrink_b1_conv"}
RETAINED = list(UNIFIED_ID)
U = "unified_selected"
RB = "resbench_resolution_only"
ARROW = "\u2192"
PROG = "16%s24%s32" % (ARROW, ARROW)
WIDTH = 6.75          # AISTATS text width, inches

# --------------------------------------------------------------------------- records

_cache: dict = {}


def git_bytes(commit: str, path: str) -> bytes:
    key = (commit, path)
    if key not in _cache:
        r = subprocess.run(["git", "-C", str(ROOT), "show", "%s:%s" % (commit, path)],
                           capture_output=True)
        if r.returncode:
            raise FileNotFoundError("%s:%s -- %s" % (commit[:10], path,
                                                     r.stderr.decode(errors="replace")))
        _cache[key] = r.stdout
    return _cache[key]


def git_json(commit: str, path: str):
    return json.loads(git_bytes(commit, path).decode("utf-8"))


def loc_commit(loc: dict) -> str:
    return loc.get("ref") or BENCH_COMMIT


def loc_str(loc: dict) -> str:
    c = loc_commit(loc)
    return "%s@%s:%s" % (REF_NAMES.get(c, "?"), c[:10], loc["path"])


IX = git_json(BENCH_COMMIT, INDEX_PATH)
EXPS = {e["id"]: e for e in IX["experiments"]}
CFGS = IX["configurations"]
CELLS = IX["cells"]


def cfgs_of(exp: str) -> list:
    return [c for c in CFGS if c["experiment"] == exp]


def cfg(exp: str, cid: str) -> dict:
    m = [c for c in cfgs_of(exp) if c["config_id"] == cid]
    if len(m) != 1:
        raise KeyError("%s/%s matches %d configurations" % (exp, cid, len(m)))
    return m[0]


def cells_for(c: dict) -> list:
    groups = set(c["run_conditions"].get("groups") or [])
    out = [x for x in CELLS if x["experiment"] == c["experiment"]
           and x["config_id"] == c["config_id"] and (not groups or x["group"] in groups)]
    return sorted(out, key=lambda x: x["seed"])


PROV: dict = {"outputs": {}}


def record(out: str, configs: list, notes: str = "", extra_files=()):
    cells = [x for c in configs for x in cells_for(c)]
    PROV["outputs"][out] = {
        "notes": notes,
        "experiments": sorted({c["experiment"] for c in configs}),
        "configurations": [{"experiment": c["experiment"], "config_id": c["config_id"],
                            "label": c["label"], "seeds_valid": c["seeds_valid"],
                            "seeds_attempted": c["seeds_attempted"]} for c in configs],
        "cells": [{"cell_id": x["cell_id"], "seed": x["seed"],
                   "summary": loc_str(x["summary"]) if x.get("summary") else None,
                   "metrics": loc_str(x["metrics"]) if x.get("metrics") else None}
                  for x in cells],
        "extra_sources": list(extra_files)}


# --------------------------------------------------------------------------- numbers

def acc_by_seed(cid: str, exp: str = U) -> dict:
    return {x["seed"]: 100 * x["final_acc"] for x in cells_for(cfg(exp, cid))
            if x["numerically_valid"]}


def paired(a: str, b: str, exp: str = U):
    A, B = acc_by_seed(a, exp), acc_by_seed(b, exp)
    seeds = sorted(set(A) & set(B))
    ca, cb = cfg(exp, a), cfg(exp, b)
    assert ca["asset_sets"] == cb["asset_sets"] and len(ca["asset_sets"]) == 1, (a, b)
    d = [A[s] - B[s] for s in seeds]
    return st.mean(d), (st.stdev(d) if len(d) > 1 else None), seeds, d


def metrics_of(cell: dict) -> list:
    return git_json(loc_commit(cell["metrics"]), cell["metrics"]["path"])


def check_against_core():
    """The index, the raw files and continuation_core.methods must agree."""
    from continuation_core.methods import METHODS
    for mid, uid in UNIFIED_ID.items():
        got = {str(s): round(v / 100, 4) for s, v in acc_by_seed(uid).items()}
        want = {k: round(v, 4) for k, v in METHODS[mid].source["final_test_acc_per_seed"].items()}
        assert got == want, (mid, got, want)
        for x in cells_for(cfg(U, uid)):
            s = git_json(loc_commit(x["summary"]), x["summary"]["path"])
            m = metrics_of(x)
            assert abs(m[-1]["test_acc_current"] - x["final_acc"]) < 1e-12, x["cell_id"]
            assert m[-1]["test_acc_current"] == m[-1]["test_acc_bypass32"], x["cell_id"]
            assert len(m) == 31 and m[-1]["epoch"] == 30 and m[-1]["update"] == 11730
            assert x["asset_set"] == "r20bn-campaign-assets"
            assert isinstance(s, dict)


def f2(v):
    return "%.2f" % v


def macro(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    return "\\newcommand{\\%s}{%s}" % (name, value)


def write_tex(name: str, body: str, sources: str):
    head = ("% GENERATED by paper/tools/make_paper_assets.py -- do not edit by hand.\n"
            "% Sources: " + sources + "\n"
            "% Provenance (configs, seeds, cells, files): paper/provenance/generated_assets.json\n")
    (TAB / name).write_text(head + body, encoding="utf-8")
    print("wrote tables/%s" % name)


def numbers():
    L = []
    means, sds = {}, {}
    for mid, uid in UNIFIED_ID.items():
        c = cfg(U, uid)
        means[mid], sds[mid] = 100 * c["acc_mean"], 100 * c["acc_sd"]
    short = {"plain": "Plain", "resolution_max_b1": "Res",
             "gaussian_postrelu": "Gauss", "resolution_max_b1_gaussian_conv": "Comb"}
    for mid in RETAINED:
        L.append(macro("u%s" % short[mid], f2(means[mid])))
        L.append(macro("u%sSD" % short[mid], f2(sds[mid])))
    for mid in RETAINED[1:]:
        m, s, _, _ = paired(UNIFIED_ID[mid], "plain")
        assert abs(m - (means[mid] - means["plain"])) < 1e-9
        L.append(macro("uGain%s" % short[mid], f2(m)))
        L.append(macro("uGain%sSD" % short[mid], f2(s)))
    for name, (a, b) in {"CombMinusResRelu": ("shrink_b1_conv", "shrink_b1_relu"),
                         "CombMinusRes": ("shrink_b1_conv", "shrink_b1"),
                         "GaussMinusBlurConv": ("blur_relu", "blur_conv")}.items():
        m, s, _, d = paired(a, b)
        assert min(d) > 0, (name, d)          # the text says "positive in each seed"
        L.append(macro("u%s" % name, f2(m)))
        L.append(macro("u%sSD" % name, f2(s)))
    for name, cid in {"BlurConv": "blur_conv", "ResRelu": "shrink_b1_relu"}.items():
        c = cfg(U, cid)
        L.append(macro("u%s" % name, f2(100 * c["acc_mean"])))
        L.append(macro("u%sSD" % name, f2(100 * c["acc_sd"])))
    for mid, uid in UNIFIED_ID.items():
        L.append(macro("uWall%s" % short[mid], "%.0f" % cfg(U, uid)["wall_seconds_mean"]))
    # final losses (record 30 of every unified run)
    for mid, uid in UNIFIED_ID.items():
        cells = cells_for(cfg(U, uid))
        last = [metrics_of(x)[-1] for x in cells]
        for key, field in (("Probe", "train_probe_ce_current"), ("TrainLoss", "train_loss_epoch"),
                           ("TestCE", "test_ce_current")):
            L.append(macro("u%s%s" % (key, short[mid]), f2(st.mean(r[field] for r in last))))
    # text of Discussion: how many other reduction+Gaussian arms lie within 0.6 pp
    comb = means["resolution_max_b1_gaussian_conv"]
    near = [c for c in cfgs_of(U) if c["config_id"].startswith("shrink_") and "_" in c["config_id"][7:]
            and c["config_id"] != "shrink_b1_conv" and comb - 100 * c["acc_mean"] < 0.6]
    L.append(macro("uNearCombCount", {6: "six"}.get(len(near), str(len(near)))))
    rb = cfg(RB, "none__none__none")
    L.append(macro("rbPlain", f2(100 * rb["acc_mean"])))
    L.append(macro("rbPlainSD", f2(100 * rb["acc_sd"])))
    L.append(macro("rbWallPlain", "%.0f" % rb["wall_seconds_mean"]))
    b1 = [c for c in cfgs_of(RB) if c["config_id"].endswith("__D1__Rprog") and c["n_valid"] > 0]
    L.append(macro("rbWallBOneMin", "%.0f" % min(c["wall_seconds_mean"] for c in b1)))
    L.append(macro("rbWallBOneMax", "%.0f" % max(c["wall_seconds_mean"] for c in b1)))
    L.append(macro("rbMaxBOne", f2(100 * cfg(RB, "max__D1__Rprog")["acc_mean"])))
    obs = {o["id"]: o for o in IX["observations"]}["same-assets-plain-differs"]
    L.append(macro("plainSpreadMax", f2(100 * max(obs["max_minus_min_by_seed"].values()))))
    L.append(macro("idxExperiments", str(len(IX["experiments"]))))
    L.append(macro("idxConfigurations", str(len(CFGS))))
    L.append(macro("idxRuns", str(len(CELLS))))
    L.append(macro("benchCommit", BENCH_COMMIT[:7]))
    write_tex("numbers.tex", "\n".join(L) + "\n",
              "%s@%s:%s; metrics.json of the unified cells" % (BENCH_BRANCH, BENCH_COMMIT[:10],
                                                                INDEX_PATH))
    record("tables/numbers.tex", [cfg(U, u) for u in
                                  ["plain", "shrink_b1", "blur_relu", "shrink_b1_conv",
                                   "shrink_b1_relu", "blur_conv"]] + near + [rb] + b1,
           notes="gains are means of per-seed paired differences (same seed = same pinned "
                 "initial weights and data order); SD is the sample SD of those differences")


# --------------------------------------------------------------------------- tables

def tables_unified():
    rows = []
    for mid, uid in UNIFIED_ID.items():
        c = cfg(U, uid)
        seeds = acc_by_seed(uid)
        if mid == "plain":
            gain = "---"
        else:
            m, s, _, _ = paired(uid, "plain")
            gain = "$%+.2f$ ($%.2f$)" % (m, s)
        rows.append("%s & $%s\\pm%s$ & %s & %s & %.0f\\\\" % (
            {"plain": "Plain", "resolution_max_b1": "\\methodR",
             "gaussian_postrelu": "\\methodG",
             "resolution_max_b1_gaussian_conv": "\\methodRG"}[mid],
            f2(100 * c["acc_mean"]), f2(100 * c["acc_sd"]),
            " / ".join(f2(seeds[s]) for s in (0, 1, 2)), gain, c["wall_seconds_mean"]))
    body = r"""\begin{table}[!htbp]
\caption{Retained procedures in the unified batch (\texttt{unified\_selected}, 48 runs). Final test accuracy after 30 epochs, where every procedure is in its target state (native resolution, no filter), so the current and target evaluations coincide. Mean $\pm$ sample SD over seeds 0--2. Same seed means the same pinned initial weights and data order, so gains are means of per-seed paired differences, with the sample SD of those differences in parentheses; neither is a confidence interval. Wall time is the mean per run, from model construction to the final summary, including per-epoch evaluation and checkpoint writes, two runs sharing a two-GPU Kaggle kernel (one T4 each).}
\label{tab:unified-reference}
\centering\small
\begin{tabular}{@{}lcccr@{}}
\toprule
Procedure & Test acc.\ (\%) & Seeds 0 / 1 / 2 & Gain over plain (pp) & Wall (s)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_tex("unified_reference.tex", body, "index configurations/cells of unified_selected")
    record("tables/unified_reference.tex", [cfg(U, u) for u in UNIFIED_ID.values()])


def tables_losses():
    names = {"plain": "Plain", "resolution_max_b1": "\\methodR",
             "gaussian_postrelu": "\\methodG", "resolution_max_b1_gaussian_conv": "\\methodRG"}
    rows = []
    for mid, uid in UNIFIED_ID.items():
        last = [metrics_of(x)[-1] for x in cells_for(cfg(U, uid))]

        def ms(field, scale=1.0):
            v = [scale * r[field] for r in last]
            return "$%s\\pm%s$" % (f2(st.mean(v)), f2(st.stdev(v)))
        rows.append("%s & %s & %s & %s & %s\\\\" % (
            names[mid], ms("test_acc_current", 100), ms("test_ce_current"),
            ms("train_probe_ce_current"), ms("train_loss_epoch")))
    body = r"""\begin{table}[!htbp]
\caption{Final losses of the retained procedures (unified batch, record after epoch 30, mean $\pm$ sample SD over three seeds). Test and training-probe quantities are evaluated in eval mode with the stored BatchNorm running statistics on the final weights. The training probe is a fixed set of 500 training images, not the full training sample. The last column is the example-weighted mean of the microbatch cross-entropies computed during the 30th epoch, in train mode, while the weights were still being updated; the cross-entropy of the final weights on the full training set was not evaluated.}
\label{tab:final-losses}
\centering\small
\begin{tabular}{@{}lcccc@{}}
\toprule
Procedure & Test acc.\ (\%) & Test CE & Training-probe CE & Last-epoch training loss\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_tex("final_losses.tex", body, "metrics.json record 30 of the unified cells")
    record("tables/final_losses.tex", [cfg(U, u) for u in UNIFIED_ID.values()])


OPS = ["max", "maxblur", "softpool", "bilinear", "l2", "hminus1", "perceptual"]
OP_TEX = {"max": "Adaptive max-pool", "maxblur": "MaxBlur (max, binomial blur)",
          "softpool": "SoftPool", "bilinear": "Bilinear (antialiased)",
          "l2": "$L^2$ reconstruction", "hminus1": "$\\dot H^{-1}$ reconstruction",
          "perceptual": "Perceptual (SSIM-style)"}
OP_MPL = {"max": "adaptive max-pool", "maxblur": "MaxBlur (max, binomial blur)",
          "softpool": "SoftPool", "bilinear": "bilinear (antialiased)",
          "l2": r"$L^2$ reconstruction", "hminus1": r"$\dot{H}^{-1}$ reconstruction",
          "perceptual": "perceptual (SSIM-style)"}


def acc_cell_tex(c):
    if c["n_valid"] == 0:
        return "Diverged (0/%d valid)" % len(c["seeds_attempted"])
    if c["acc_sd"] is None:
        return "%s (1 seed)" % f2(100 * c["acc_mean"])
    return "$%s\\pm%s$" % (f2(100 * c["acc_mean"]), f2(100 * c["acc_sd"]))


def tables_operators():
    rows, used = [], []
    for op in OPS:
        a, b = cfg(RB, "%s__D1__Rprog" % op), cfg(RB, "%s__input__Rprog" % op)
        used += [a, b]
        rows.append("%s & %s & %s\\\\" % (OP_TEX[op], acc_cell_tex(a), acc_cell_tex(b)))
    base = cfg(RB, "none__none__none")
    used.append(base)
    cap = (r"\caption{Resolution-only benchmark (\texttt{resbench\_resolution\_only}): reduction "
           r"operator at two locations, schedule $16\to24\to32$, no Gaussian. Final test accuracy "
           r"after 30 epochs on the native path, mean $\pm$ sample SD over three seeds. The "
           r"batch's own plain arm reaches $" + f2(100 * base["acc_mean"]) + r"\pm"
           + f2(100 * base["acc_sd"]) + r"\%$. A run whose loss became non-finite is a failed "
           r"numerical experiment: the perceptual operator after block 1 diverged in all three "
           r"seeds and has no accuracy.}")
    body = r"""\begin{table}[!htbp]
""" + cap + r"""
\label{tab:operators}
\centering\small
\begin{tabular}{@{}lcc@{}}
\toprule
Reduction operator & After block 1 (\%) & At the input (\%)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_tex("resolution_operators.tex", body, "index configurations of resbench_resolution_only")
    record("tables/resolution_operators.tex", used)


def tables_plain_arms():
    obs = {o["id"]: o for o in IX["observations"]}["same-assets-plain-differs"]
    rows = []
    for exp, per in obs["plain_final_acc_by_experiment_and_seed"].items():
        v = [100 * per[s] for s in ("0", "1", "2")]
        rows.append("\\texttt{%s} & %s & %s & %s\\\\" % (
            tex(exp), f2(v[0]), f2(v[1]), f2(v[2])))
    spread = ["%s" % f2(100 * obs["max_minus_min_by_seed"][s]) for s in ("0", "1", "2")]
    body = r"""\begin{table}[!htbp]
\caption{Plain ResNet-20 arms that verified identical pinned assets (initial weights, BatchNorm buffers, training probe, per-epoch order) in four batches. Final test accuracy (\%). Evaluations before training are identical; the trajectories differ from the first later record. The cause is not established, so no correction is applied between batches.}
\label{tab:plain-arms}
\centering\small
\begin{tabular}{@{}lccc@{}}
\toprule
Experiment & Seed 0 & Seed 1 & Seed 2\\
\midrule
""" + "\n".join(rows) + "\n\\midrule\nMax $-$ min & %s & %s & %s\\\\\n" % tuple(spread) + r"""\bottomrule
\end{tabular}
\end{table}
"""
    write_tex("plain_arms.tex", body, "index observation same-assets-plain-differs")
    record("tables/plain_arms.tex",
           [cfg(e, cid) for e, cid in (("fulldata_gaussian", "plain"),
                                       ("campaign_grid21", "R32__Gnone__input_bilinear__all19"),
                                       (RB, "none__none__none"), (U, "plain"))])


def tables_assets():
    rows = []
    for name, s in IX["asset_sets"].items():
        a, sts = s["arrays"], s["states"]
        rows.append("\\texttt{%s} & \\texttt{%s} & \\texttt{%s} & \\texttt{%s} / \\texttt{%s} / \\texttt{%s} & \\texttt{%s} / \\texttt{%s} / \\texttt{%s}\\\\" % (
            tex(name), a["subset"][:8], a["train_probe"][:8], a["perm_seed0"][:8],
            a["perm_seed1"][:8], a["perm_seed2"][:8], sts["init_seed0"][:8],
            sts["init_seed1"][:8], sts["init_seed2"][:8]))
    body = r"""\begin{table}[!htbp]
\caption{Pinned asset sets, identified by SHA-256 prefixes of their content. Pairing between runs is decided by these digests, not by batch names.}
\label{tab:assets}
\centering\scriptsize
\begin{tabular}{@{}lcccc@{}}
\toprule
Asset set & Subset & Probe & Order, seeds 0 / 1 / 2 & Initial state, seeds 0 / 1 / 2\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_tex("asset_sets.tex", body, "index asset_sets")
    PROV["outputs"]["tables/asset_sets.tex"] = {"extra_sources": [
        "%s@%s:%s#asset_sets" % (BENCH_BRANCH, BENCH_COMMIT[:10], INDEX_PATH)]}


def tables_schedule():
    from continuation_core.controller import InterventionController
    from continuation_core.methods import METHODS, PLATEAU_G, RPROG
    from continuation_core.models import site_map
    sm = site_map("resnet20_bn_cifar")
    ctrl = {m: InterventionController(METHODS[m], sm, "resnet20_bn_cifar") for m in RETAINED}
    bounds = sorted(set(PLATEAU_G.starts) | set(RPROG.starts)) + [30]
    rows = []
    for e0, e1 in zip(bounds, bounds[1:]):
        sig = {}
        for m, c in ctrl.items():
            c.set_epoch(e0)
            vals = c.per_site_sigma()
            for e in range(e0, e1):          # constant over the whole range
                c.set_epoch(e)
                assert c.per_site_sigma() == vals
            assert len(set(vals)) <= 1
            sig[m] = vals[0] if vals else None
        r, g = RPROG.at(e0), PLATEAU_G.at(e0)

        def s(v):
            return "0 (bypass)" if v == 0 else "%.3f" % v
        rows.append("%d--%d & %d & %s & %.2f & %s & %s\\\\" % (
            e0, e1 - 1, 391 * e0, "32 (bypass)" if r == 32 else str(r), g,
            s(sig["gaussian_postrelu"]), s(sig["resolution_max_b1_gaussian_conv"])))
    assert len(ctrl["gaussian_postrelu"].per_site_sigma()) == 10
    assert len(ctrl["resolution_max_b1_gaussian_conv"].per_site_sigma()) == 19
    body = r"""\begin{table}[!htbp]
\caption{Executed schedules, read from the frozen \texttt{continuation-core} presets and evaluated through their controller. Epochs are zero-based and a state holds for a whole epoch of 391 updates, so the first update of epoch $e$ has index $391e$. $r$ is the side of the map entering residual block index 2 in \methodR{} and \methodRG; $G$ is the reference Gaussian level. Effective $\sigma$ is in pixels of the feature map at the filtered site and, within a procedure, is the same at every site. \methodG{} uses $G$ only and never changes resolution.}
\label{tab:schedules}
\centering\small
\begin{tabular}{@{}rrcccc@{}}
\toprule
Epochs & First update & $r$ & $G$ & \methodG: $\sigma$ (10 sites) & \methodRG: $\sigma=\frac{r}{32}G$ (19 sites)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_tex("schedules.tex", body, "continuation_core.methods (PLATEAU_G, RPROG, METHODS) "
              "and continuation_core.controller.InterventionController.per_site_sigma")
    PROV["outputs"]["tables/schedules.tex"] = {"extra_sources": [
        "continuation_core/methods.py", "continuation_core/controller.py",
        "continuation_core/models/resnet20_bn.py"]}


# --------------------------------------------------------------------------- LaTeX helpers

_UNI = {"\u2192": "$\\to$", "\u2014": "---", "\u2013": "--", "\u00d7": "$\\times$",
        "\u03c3": "$\\sigma$", "\u2212": "$-$", "\u2248": "$\\approx$"}


def tex(s) -> str:
    s = str(s).replace("H^-1", "\x01HM\x01")
    s = s.replace("\\", "\\textbackslash{}")
    for a, b in (("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"),
                 ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"),
                 ("^", "\\textasciicircum{}")):
        s = s.replace(a, b)
    s = s.replace("\\textbackslash\\{\\}", "\\textbackslash{}")
    s = s.replace("\\textasciitilde\\{\\}", "\\textasciitilde{}")
    s = s.replace("\\textasciicircum\\{\\}", "\\textasciicircum{}")
    for a, b in _UNI.items():
        s = s.replace(a, b)
    s = s.replace("\x01HM\x01", "$H^{-1}$")
    bad = [ch for ch in s if ord(ch) > 127]
    if bad:
        raise ValueError("non-ASCII in LaTeX text: %r" % "".join(bad))
    return s


def tt_id(s: str) -> str:
    return "\\texttt{%s}" % tex(s).replace("\\_", "\\_\\allowbreak{}")


def short_label(label: str) -> str:
    return re.sub(r"\[launch ([A-Za-z0-9-]+?)-\d{8}-\d{6}\]", r"[launch \1]", label)


# --------------------------------------------------------------------------- complete table

ORDER = ["unified_selected", "resbench_resolution_only", "campaign_grid21", "ablation_aa",
         "per_layer_sigma", "adaptive_continuation", "fulldata_gaussian",
         "progressive_resolution_pilot", "resnet20bn_gaussian_pilot",
         "resnet18bn_gaussian_pilot", "db2_pilot", "gn_lr_audit", "pilot_internal_gaussian",
         "exp0_input_gaussian", "exp1_input_warmstart"]
NO_TRAINING = ["tv_budget_previews", "wavelet_previews", "loss_landscape_blur",
               "infrastructure_probes"]
PAIRING = {
    "r20bn-campaign-assets": "shares initial weights, BatchNorm buffers, probe and per-epoch "
                             "order, seed by seed, with every experiment on this set",
    "r20bn-ablation-assets": "same subset, probe and per-epoch order as r20bn-campaign-assets, "
                             "different initial weights",
    "per-layer-self-paired": "self-paired within the study; its digests use another scheme, "
                             "so pairing with the other sets is not established",
    "None": "no pinned assets recorded",
}
ENDPOINT = {"current=target": "target", "current": "current (control)",
            "target (declared primary_path)": "target", "target": "target",
            "current (declared primary_path)": "current (control)",
            "target (unfiltered val images)": "target", "target (unfiltered)": "target"}
RAG = ">{\\raggedright\\arraybackslash}"


def complete_table():
    L = [r"""\begingroup\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}""" + RAG + "p{2.5in}" + RAG + "p{1.38in}rrc" + RAG + r"""p{0.62in}r@{}}
\caption{Complete audited record of trained configurations, grouped by experiment. One row per configuration and run conditions; reruns under a separate launch are separate rows, while byte-identical copies of the same summary are collapsed. Accuracy: final record of each run, mean over numerically valid seeds, no best-epoch selection. SD: sample SD in points; \emph{n/a} when only one valid seed exists. Seeds: valid / attempted. Endpoint: \emph{target} is the native, filter-free path, which coincides with the current path at the end of every continuation arm; \emph{current (control)} marks arms whose own intervention remains at inference. Wall: mean seconds per run, with the timing scope given in the experiment header. Rows within an experiment are listed alphabetically, not ranked; comparisons across experiments are not paired unless the asset sets match and the run conditions are the same.}\label{tab:all-exploratory}\\
\toprule
Configuration & Config id & Acc.\ (\%) & SD & Seeds & Endpoint & Wall (s)\\
\midrule
\endfirsthead
\toprule
Configuration & Config id & Acc.\ (\%) & SD & Seeds & Endpoint & Wall (s)\\
\midrule
\endhead
\midrule\multicolumn{7}{r@{}}{\emph{continued on next page}}\\
\endfoot
\bottomrule
\endlastfoot
"""]
    used = []
    for eid in ORDER:
        e = EXPS[eid]
        rows = cfgs_of(eid)
        used += rows
        cnt = e["counts"]
        refs = sorted({loc_commit(g["location"]) for g in e["groups"]})
        where = "; ".join("%s@%s" % (REF_NAMES.get(r, "?"), r[:7]) for r in refs)
        code_commits = sorted({(g.get("introduced_by_commit") or "")[:7] for g in e["groups"]} - {""})
        shipped = [g["shipped_code_check"] for g in e["groups"] if g.get("shipped_code_check")]
        ship = ""
        if shipped:
            ship = " Shipped code checked against commit %s." % shipped[0]["compared_with_commit"][:7]
        dups = sum(g.get("duplicate_summaries_collapsed") or 0 for g in e["groups"])
        splits = sorted({c["split"] for c in rows if c.get("split")})
        assets = "; ".join("\\texttt{%s}: %s" % (tex(a), tex(PAIRING.get(a, a)))
                           for a in e["asset_sets"])
        hdr = ("\\multicolumn{7}{@{}" + RAG + "p{\\linewidth}@{}}{\\rule{0pt}{2.6ex}\\textbf{%s} "
               "(\\texttt{%s}; owner %s; %s)}\\\\*\n") % (
                   tex(e["title"]), tex(eid), tex(e["owner"]), tex(e["kind"].replace("_", " ")))
        info = ("Conditions: %s. Evaluation split: %s. Records: %s; results committed in %s.%s "
                "Knowledge-base record: %s. Assets: %s. Runs attempted per manifests: %s; with "
                "summary: %d; numerically valid: %d; diverged: %d; failure markers: %d; "
                "duplicate summaries collapsed: %d. Timing: %s" % (
                    tex(e["conditions"].rstrip(".")), tex(", ".join(splits) or "n/a"), tex(where),
                    tex(", ".join(code_commits) or "n/a"), tex(ship),
                    tex(e.get("knowledge_base_record") or "none"), assets or "none",
                    cnt["cells_attempted_per_manifests"] if cnt["cells_attempted_per_manifests"]
                    is not None else "not recorded",
                    cnt["cells_with_summary"], cnt["cells_numerically_valid"],
                    cnt["cells_diverged"], cnt["failure_markers"], dups,
                    tex(e["timing_scope"])))
        L.append(hdr)
        L.append(("\\multicolumn{7}{@{}" + RAG + "p{\\linewidth}@{}}{\\scriptsize %s}\\\\*[2pt]\n")
                 % info)
        rows = sorted(rows, key=lambda c: (c["label"].lower(), c["config_id"]))
        for c in rows:
            if c["n_valid"] == 0:
                acc, sd = "---", "---"
            else:
                acc = f2(100 * c["acc_mean"])
                sd = f2(100 * c["acc_sd"]) if c["acc_sd"] is not None else "n/a"
            lab = tex(short_label(c["label"]))
            if c["status"] != "valid":
                lab += " \\textbf{[%s: non-finite loss, seeds %s]}" % (
                    tex(c["status"]), tex(", ".join(map(str, c["seeds_diverged"]))))
            for fl in c["flags"]:
                lab += " \\emph{(%s)}" % tex(fl)
            fp = c["final_path"] if isinstance(c["final_path"], str) else ", ".join(c["final_path"])
            wall = "%.0f" % c["wall_seconds_mean"] if c["wall_seconds_mean"] else "---"
            L.append("%s & %s & %s & %s & %d/%d & %s & %s\\\\\n" % (
                lab, tt_id(c["config_id"]), acc, sd, c["n_valid"], len(c["seeds_attempted"]),
                tex(ENDPOINT.get(fp, fp)), wall))
    L.append("\\midrule\n\\multicolumn{7}{@{}" + RAG + "p{\\linewidth}@{}}{\\rule{0pt}{2.6ex}"
             "\\textbf{Records without training}}\\\\*\n")
    for eid in NO_TRAINING:
        e = EXPS[eid]
        L.append(("\\multicolumn{7}{@{}" + RAG + "p{\\linewidth}@{}}{\\scriptsize \\textbf{%s} "
                  "(\\texttt{%s}; owner %s): %s.}\\\\\n") % (
                      tex(e["title"]), tex(eid), tex(e["owner"]), tex(e["conditions"].rstrip("."))))
    L.append("\\end{longtable}\n\\endgroup\n")
    write_tex("all_exploratory.tex", "".join(L),
              "%s@%s:%s (experiments, configurations)" % (BENCH_BRANCH, BENCH_COMMIT[:10],
                                                          INDEX_PATH))
    record("tables/all_exploratory.tex", used, notes="all configurations of the index")


# --------------------------------------------------------------------------- figures

ROW_H = 0.158          # inches per row in dot plots
TOP_IN, BOTTOM_IN = 0.26, 0.42


def margins(fig_h, left):
    return dict(left=left, right=0.985, top=1 - TOP_IN / fig_h, bottom=BOTTOM_IN / fig_h)


def row(c, label, color=GREY, control=None):
    seeds = [100 * v for k, v in sorted(c["acc_per_seed"].items()) if int(k) in c["seeds_valid"]]
    return {"label": label, "seeds": seeds,
            "mean": None if c["acc_mean"] is None else 100 * c["acc_mean"],
            "sd": None if c["acc_sd"] is None else 100 * c["acc_sd"],
            "n_valid": c["n_valid"], "n_att": len(c["seeds_attempted"]),
            "color": color, "control": bool(c["flags"]) if control is None else control,
            "cfg": c}


def header(text):
    return {"header": text}


def dot_panel(ax, rows, base=None, xlim=None, show_labels=True, value="acc"):
    n = len(rows)
    ys = list(range(n))[::-1]
    labels, colors, bold = [], [], []
    for y, r in zip(ys, rows):
        if "header" in r:
            ax.axhspan(y - 0.5, y + 0.5, color="0.93", zorder=0, lw=0)
    if base is not None:
        ax.axvline(base, color="#000000", lw=0.6, zorder=1)
    for y, r in zip(ys, rows):
        if "header" in r:
            labels.append(r["header"]); colors.append("#000000"); bold.append(True)
            continue
        labels.append(r["label"]); colors.append(r["color"] if r["color"] != GREY else "#333333")
        bold.append(False)
        col = r["color"]
        if value == "acc":
            m, sd, pts = r["mean"], r["sd"], r["seeds"]
        else:
            m, sd, pts = r.get("dmean"), r.get("dsd"), r.get("dpts", [])
        if m is None:
            if r.get("note"):
                ax.text(0.015, y, r["note"], transform=ax.get_yaxis_transform(), va="center",
                        fontsize=6.3, color="#a00000" if r["n_valid"] == 0 else "0.35")
            continue
        if sd is not None:
            ax.plot([m - sd, m + sd], [y, y], "-", color=col, lw=2.2, alpha=0.35, zorder=2,
                    solid_capstyle="butt")
        for v in pts:
            ax.plot([v], [y], "o", ms=2.4, mfc="none", mec=col, mew=0.6, zorder=3)
        ax.plot([m], [y], "s" if r["control"] else "o", ms=4.2,
                mfc="white" if r["control"] else col, mec=col, mew=0.9, zorder=4)
        if value == "acc" and r["n_valid"] == 1:
            ax.annotate("1 seed", (m, y), xytext=(4, 0), textcoords="offset points",
                        fontsize=5.8, color="0.45", va="center")
    ax.set_yticks(ys)
    if show_labels:
        ax.set_yticklabels(labels)
        for t, c, b in zip(ax.get_yticklabels(), colors, bold):
            t.set_color(c)
            if b:
                t.set_fontweight("bold")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, n - 0.3)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", alpha=0.25, lw=0.5)


def add_legend(fig, sd=True, control=False, base=True):
    h = [Line2D([], [], ls="", marker="o", ms=2.4, mfc="none", mec="0.3", mew=0.6,
                label="individual seed"),
         Line2D([], [], ls="", marker="o", ms=4.2, color="0.3", label="mean over valid seeds")]
    if sd:
        h.append(Line2D([], [], color="0.3", lw=2.2, alpha=0.35, label="$\\pm1$ sample SD"))
    if control:
        h.append(Line2D([], [], ls="", marker="s", ms=4.2, mfc="white", mec="0.3", mew=0.9,
                        label="control with a different endpoint"))
    if base:
        h.append(Line2D([], [], color="#000000", lw=0.6, label="plain mean of the same batch"))
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(h),
               frameon=False, handlelength=1.6, columnspacing=1.4)


def save(fig, name, configs, notes, extra=()):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, metadata={"Creator": COMMAND, "Subject": notes[:200]})
    plt.close(fig)
    record("figures/" + name, configs, notes=notes, extra_files=extra)
    print("wrote figures/%s" % name)


def fig_unified():
    U_LABEL = {
        "plain": "plain (no intervention)",
        "shrink_input": "input image, bilinear",
        "shrink_b0": "max-pool after block 0",
        "shrink_b1": "max-pool after block 1 = Resolution",
        "shrink_b2": "max-pool after block 2",
        "blur_relu": "10 post-ReLU sites = Gaussian",
        "blur_conv": "19 convolution outputs",
        "shrink_input_relu": "input bilinear + post-ReLU",
        "shrink_b0_relu": "block 0 + post-ReLU",
        "shrink_b0_relu_rf": "block 0 + post-ReLU, receptive-field profile",
        "shrink_b1_relu": "block 1 + post-ReLU",
        "shrink_b1_relu_rf": "block 1 + post-ReLU, receptive-field profile",
        "shrink_b1_relu_sqrt": "block 1 + post-ReLU, sqrt(map size) profile",
        "shrink_b1_conv": "block 1 + 19 conv outputs = Combined",
        "shrink_b2_relu": "block 2 + post-ReLU",
        "shrink_b2_relu_rf": "block 2 + post-ReLU, receptive-field profile",
    }
    layout = [header("Control"), "plain",
              header("Resolution only, " + PROG), "shrink_input", "shrink_b0", "shrink_b1",
              "shrink_b2",
              header("Gaussian only, annealed to 0 at epoch 21"), "blur_relu", "blur_conv",
              header("Resolution (max-pool unless noted) + Gaussian"), "shrink_input_relu",
              "shrink_b0_relu", "shrink_b0_relu_rf", "shrink_b1_relu", "shrink_b1_relu_rf",
              "shrink_b1_relu_sqrt", "shrink_b1_conv", "shrink_b2_relu", "shrink_b2_relu_rf"]
    inv = {v: k for k, v in UNIFIED_ID.items()}
    assert {x for x in layout if isinstance(x, str)} == {c["config_id"] for c in cfgs_of(U)}
    rows, used = [], []
    for x in layout:
        if isinstance(x, dict):
            rows.append(x)
            continue
        c = cfg(U, x)
        used.append(c)
        r = row(c, U_LABEL[x], COLOR.get(inv.get(x), GREY), control=False)
        if x != "plain":
            m, s, _, d = paired(x, "plain")
            r.update(dmean=m, dsd=s, dpts=d)
        rows.append(r)
    h = ROW_H * len(rows) + TOP_IN + BOTTOM_IN
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, h), sharey=True,
                               gridspec_kw={"width_ratios": [1.0, 0.8], "wspace": 0.04,
                                            **margins(h, 0.35)})
    base = 100 * cfg(U, "plain")["acc_mean"]
    dot_panel(a, rows, base=base, xlim=(73.8, 82.6))
    a.set_xlabel("final test accuracy after 30 epochs (%)")
    a.set_title("(a) Final accuracy, native filter-free path", loc="left")
    dot_panel(b, rows, base=0.0, xlim=(2.2, 7.4), show_labels=False, value="diff")
    b.set_xlabel("difference to plain, same seed (pp)")
    b.set_title("(b) Per-seed paired difference", loc="left")
    add_legend(fig)
    save(fig, "appendix_unified.pdf", used,
         "unified_selected, all 16 configurations x seeds 0-2; (a) final test accuracy, "
         "record 30, current=target; (b) per-seed differences to plain on identical assets")


def seed_curves(uid, field, scale=1.0):
    per = []
    for x in cells_for(cfg(U, uid)):
        m = metrics_of(x)
        per.append({r["epoch"]: scale * r[field] for r in m if r.get(field) is not None})
    ep = sorted(set.intersection(*[set(p) for p in per]))
    arr = np.array([[p[e] for e in ep] for p in per])
    return np.array(ep), arr


def fig_unified_curves():
    panels = [("test_acc_current", 100, "(a) Test accuracy, current path (intervention active)",
               "test accuracy (%)"),
              ("test_acc_bypass32", 100, "(b) Test accuracy, target path (diagnostic)",
               "test accuracy (%)"),
              ("train_probe_ce_current", 1, "(c) Training-probe CE (500 images), current path",
               "cross-entropy (nats)"),
              ("test_ce_current", 1, "(d) Test CE, current path", "cross-entropy (nats)")]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.5), sharex=True)
    for ax, (field, scale, title, ylab) in zip(axes.flat, panels):
        for x0, txt in ((6.5, "r: 16%s24" % ARROW), (12.5, "r: 24%s32" % ARROW),
                        (21.5, "G %s 0" % ARROW)):
            ax.axvline(x0, color="0.75", lw=0.6, zorder=0)
            if field == "test_acc_current":
                ax.text(x0, 1.01, txt, transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=6, color="0.4")
        for mid in RETAINED:
            ep, arr = seed_curves(UNIFIED_ID[mid], field, scale)
            assert len(ep) == 31 and arr.shape[0] == 3, (mid, field, arr.shape)
            m, s = arr.mean(0), arr.std(0, ddof=1)
            ax.fill_between(ep, m - s, m + s, color=COLOR[mid], alpha=0.16, lw=0)
            ax.plot(ep, m, "-", color=COLOR[mid], lw=1.1, label=PAPER_NAME[mid])
        ax.set_title(title, loc="left", pad=11 if field == "test_acc_current" else 4)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(0, 30)
    axes[0, 0].set_ylim(10, 85)
    axes[0, 1].set_ylim(0, 85)
    axes[1, 0].set_ylim(0, 2.4)
    axes[1, 1].set_ylim(0.4, 2.4)
    for ax in axes[1]:
        ax.set_xlabel("completed epochs $k$")
    h = [Line2D([], [], color=COLOR[m], lw=1.1, label=PAPER_NAME[m]) for m in RETAINED]
    h.append(Line2D([], [], color="0.3", lw=5, alpha=0.16, label="$\\pm1$ sample SD (3 seeds)"))
    axes[0, 0].legend(handles=h, loc="lower right", frameon=True, edgecolor="0.8")
    fig.tight_layout(h_pad=0.8, w_pad=1.2)
    extra = sorted({loc_str(x["metrics"]) for u in UNIFIED_ID.values()
                    for x in cells_for(cfg(U, u))})
    save(fig, "appendix_unified_curves.pdf", [cfg(U, u) for u in UNIFIED_ID.values()],
         "unified_selected, retained procedures, metrics.json records 0-30 of seeds 0-2; "
         "current = state of the epoch just completed; target = bypass32 evaluation", extra)


def fig_resolution_operators():
    base = 100 * cfg(RB, "none__none__none")["acc_mean"]
    h = ROW_H * 8 + TOP_IN + BOTTOM_IN
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, h), sharey=True,
                             gridspec_kw={"wspace": 0.04, **margins(h, 0.24)})
    used = [cfg(RB, "none__none__none")]
    for ax, loc, title in ((axes[0], "D1", "(a) After residual block 1"),
                           (axes[1], "input", "(b) At the input image")):
        rows = [row(cfg(RB, "none__none__none"), "plain (this batch)", "#000000")]
        for op in OPS:
            c = cfg(RB, "%s__%s__Rprog" % (op, loc))
            used.append(c)
            r = row(c, OP_MPL[op])
            if c["n_valid"] == 0:
                r["note"] = "diverged: non-finite loss in %d / %d seeds" % (
                    len(c["seeds_diverged"]), r["n_att"])
            rows.append(r)
        dot_panel(ax, rows, base=base, xlim=(74.6, 81.6), show_labels=(loc == "D1"))
        ax.set_title(title, loc="left")
        ax.set_xlabel("final test accuracy (%)")
    add_legend(fig)
    save(fig, "appendix_resolution_operators.pdf", used,
         "resbench_resolution_only, operator x location, schedule 16-24-32, seeds 0-2")


def fig_resolution_location_schedule():
    base = 100 * cfg(RB, "none__none__none")["acc_mean"]

    def plain():
        return row(cfg(RB, "none__none__none"), "plain (this batch)", "#000000")
    loc_rows = [plain()] + [row(cfg(RB, "max__%s__Rprog" % l), n) for l, n in
                            (("input", "input image"), ("stem", "stem output"),
                             ("D0", "after block 0"), ("D1", "after block 1"),
                             ("D2", "after block 2"))]
    sched = [("Rprog", "%s, epochs 0-5 / 6-11 / 12-29" % PROG),
             ("Rlate", "%s, epochs 0-8 / 9-17 / 18-29" % PROG),
             ("Rgentle", "24%s32 only" % ARROW),
             ("Rreverse", "24%s16%s32 (order control)" % (ARROW, ARROW))]
    s_rows = [plain()]
    for op in ("max", "maxblur", "hminus1"):
        s_rows.append(header(OP_MPL[op]))
        s_rows += [row(cfg(RB, "%s__D1__%s" % (op, p)), n) for p, n in sched]
    ctl_rows = [plain(), row(cfg(RB, "max__D1__Rprog"), "%s, restored" % PROG),
                row(cfg(RB, "max__D1__fixed24"), "fixed 24\u00d724, never restored"),
                row(cfg(RB, "max__D1__fixed16"), "fixed 16\u00d716, never restored")]
    sizes = [len(loc_rows), len(s_rows), len(ctl_rows)]
    h = ROW_H * sum(sizes) + 2 * 0.45 + TOP_IN + BOTTOM_IN
    fig = plt.figure(figsize=(WIDTH, h))
    gs = fig.add_gridspec(3, 1, height_ratios=sizes, hspace=0.45 * 3 / (ROW_H * sum(sizes)),
                          **margins(h, 0.3))
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    for ax, rows, title in zip(axes, (loc_rows, s_rows, ctl_rows),
                               ("(a) Max-pool location, " + PROG,
                                "(b) Schedule, operator after block 1",
                                "(c) Restored vs fixed resolution after block 1")):
        dot_panel(ax, rows, base=base, xlim=(72.4, 81.4))
        ax.set_title(title, loc="left")
    axes[2].set_xlabel("final test accuracy after 30 epochs (%)")
    add_legend(fig, control=True)
    used = [r["cfg"] for r in loc_rows + s_rows + ctl_rows if "cfg" in r]
    uniq = list({(c["experiment"], c["config_id"]): c for c in used}.values())
    save(fig, "appendix_resolution_location_schedule.pdf", uniq,
         "resbench_resolution_only: location (max-pool), schedule (max, maxblur, hminus1 at "
         "D1), fixed-resolution controls whose final metric is the reduced path")


def campaign_label(cid: str) -> str:
    res, g, red, mask = cid.split("__")
    RES = {"Rprog": PROG, "Rgentle": "24%s32" % ARROW, "Rreverse": "24%s16%s32" % (ARROW, ARROW)}
    RED = {"input_bilinear": "input, bilinear", "input_max": "input, max-pool",
           "stem_bilinear": "stem, bilinear", "stem_max": "stem, max-pool"}
    G = {"Gplateau": "Gaussian plateaus", "Ggeo": "Gaussian, geometric decay",
         "Gmix": "blend of each map with its blur"}
    M = {"all19": "", "early7": ", first 7 layers"}
    if res == "R32" and g == "Gnone":
        return "plain (no intervention)"
    if res == "R32":
        return "%s%s" % (G[g], M[mask])
    if g == "Gnone":
        return "%s, %s" % (RED[red], RES[res])
    return "%s, %s + %s%s" % (RED[red], RES[res], G[g], M[mask])


def stacked(rows_a, rows_b, left, gap_in=0.45):
    h = ROW_H * (len(rows_a) + len(rows_b)) + gap_in + TOP_IN + BOTTOM_IN
    fig = plt.figure(figsize=(WIDTH, h))
    gs = fig.add_gridspec(2, 1, height_ratios=[len(rows_a), len(rows_b)],
                          hspace=gap_in * 2 / (ROW_H * (len(rows_a) + len(rows_b))),
                          **margins(h, left))
    a = fig.add_subplot(gs[0])
    b = fig.add_subplot(gs[1], sharex=a)
    return fig, a, b


def fig_campaign():
    E = "campaign_grid21"
    groups = [("Control", ["R32__Gnone__input_bilinear__all19"]),
              ("Gaussian at conv outputs, native resolution",
               ["R32__Gplateau__input_bilinear__all19", "R32__Ggeo__input_bilinear__all19",
                "R32__Gmix__input_bilinear__all19", "R32__Gplateau__input_bilinear__early7"]),
              ("Resolution only (input image or stem)",
               ["Rprog__Gnone__input_bilinear__all19", "Rprog__Gnone__input_max__all19",
                "Rprog__Gnone__stem_bilinear__all19", "Rprog__Gnone__stem_max__all19",
                "Rgentle__Gnone__input_bilinear__all19", "Rreverse__Gnone__input_bilinear__all19"]),
              ("Resolution + Gaussian at conv outputs",
               ["Rprog__Gplateau__input_bilinear__all19", "Rprog__Gplateau__input_max__all19",
                "Rprog__Gplateau__stem_bilinear__all19", "Rprog__Gplateau__stem_max__all19",
                "Rprog__Gplateau__input_bilinear__early7", "Rprog__Ggeo__input_bilinear__all19",
                "Rprog__Gmix__input_bilinear__all19", "Rgentle__Gplateau__input_bilinear__all19",
                "Rgentle__Ggeo__input_bilinear__all19", "Rreverse__Gplateau__input_bilinear__all19"])]
    assert sum(len(g[1]) for g in groups) == len(cfgs_of(E))
    rows, used = [], []
    for title, ids in groups:
        rows.append(header(title))
        for cid in ids:
            c = cfg(E, cid)
            used.append(c)
            rows.append(row(c, campaign_label(cid),
                            "#000000" if cid == "R32__Gnone__input_bilinear__all19" else GREY))
    f_rows = [header("fulldata_gaussian (3 seeds)")]
    for cid, lab in (("plain", "plain (no intervention)"),
                     ("plateau", "Gaussian, plateau schedule"),
                     ("geometric", "Gaussian, geometric schedule")):
        c = cfg("fulldata_gaussian", cid)
        used.append(c)
        f_rows.append(row(c, lab, "#000000" if cid == "plain" else GREY))
    f_rows.append(header("progressive_resolution_pilot (seed 0)"))
    for cid, lab in (("progres", "progressive input resolution"),
                     ("progres_gauss", "progressive input resolution + Gaussian")):
        c = cfg("progressive_resolution_pilot", cid)
        used.append(c)
        f_rows.append(row(c, lab))
    fig, a, b = stacked(rows, f_rows, 0.42)
    dot_panel(a, rows, base=100 * cfg(E, "R32__Gnone__input_bilinear__all19")["acc_mean"],
              xlim=(73.8, 82.2))
    a.set_title("(a) campaign_grid21", loc="left")
    dot_panel(b, f_rows, base=100 * cfg("fulldata_gaussian", "plain")["acc_mean"],
              xlim=(73.8, 82.2))
    b.set_title("(b) Earlier full-data runs, same assets", loc="left")
    b.set_xlabel("final test accuracy after 30 epochs (%)")
    add_legend(fig)
    save(fig, "appendix_campaign.pdf", used,
         "campaign_grid21 (21 configurations x 3 seeds), fulldata_gaussian (3 x 3), "
         "progressive_resolution_pilot (2 x seed 0); panel (b) baseline line = fulldata plain")


# Short labels for ablation_aa, written from the audited index labels.
ABL_LABEL = {
    "C_plain": "plain (no intervention)",
    "P_postbn": "Gaussian after every BatchNorm",
    "P_postblock": "Gaussian at the 10 post-ReLU sites",
    "C_plateau": "Gaussian at every conv output",
    "M_predown": "Gaussian only at the 2 layers feeding a downsample",
    "M_nodown": "Gaussian at the 17 layers away from a downsample",
    "D0": "max-pool after block 0", "D1": "max-pool after block 1",
    "D2": "max-pool after block 2",
    "R4": "first conv layer (stem) reduced", "R1": "input image reduced",
    "R5": "reduction after the first ReLU + post-ReLU Gaussian",
    "D0G": "block 0 + post-ReLU Gaussian", "D1G": "block 1 + post-ReLU Gaussian",
    "D2G": "block 2 + post-ReLU Gaussian",
    "R3": "input image + post-ReLU Gaussian", "R2": "input image + Gaussian at every conv output",
    "Q_A1": "post-ReLU, \u03c3 scaled by map size",
    "Q_A2": "post-ReLU, \u03c3 scaled by sqrt(map size)",
    "Q_A3": "post-ReLU, \u03c3 rising with depth (direction control)",
    "Q_A4": "post-ReLU, \u03c3 scaled by receptive field",
    "P_A1": "block 2 + post-ReLU, \u03c3 scaled by map size",
    "P_A2": "block 2 + post-ReLU, \u03c3 scaled by sqrt(map size)",
    "P_A3": "block 2 + post-ReLU, \u03c3 rising with depth (direction control)",
    "P_A4": "block 2 + post-ReLU, \u03c3 scaled by receptive field",
    "B_blurpool": "fixed BlurPool before each downsample",
    "B_blurpool_plateau": "fixed BlurPool + annealed Gaussian everywhere",
    "K_const030": "fixed Gaussian 0.30, never annealed",
    "K_const050": "fixed Gaussian 0.50, never annealed",
    "K_const080": "fixed Gaussian 0.80, never annealed",
    "K_const100": "fixed Gaussian 1.00, never annealed",
    "R6": "fixed post-ReLU Gaussian, never annealed",
}


def fig_ablation():
    E = "ablation_aa"
    all_c = cfgs_of(E)
    assert set(ABL_LABEL) == {c["config_id"] for c in all_c}
    fam_order = [("Control", ["plain"]),
                 ("Gaussian placement and site masks, annealed", ["gauss"]),
                 ("Resolution only, " + PROG, ["res"]),
                 ("Resolution + Gaussian", ["res_gauss"]),
                 ("Per-layer \u03c3 profiles", ["profile"]),
                 ("Controls with a different endpoint", ["constant"])]
    rows, used = [], []
    placed = set()
    for title, fams in fam_order:
        grp = []
        for c in all_c:
            is_ctl = bool(c["flags"])
            ok = is_ctl if title.startswith("Controls") else (c["family"] in fams and not is_ctl)
            if ok and c["config_id"] not in placed:
                grp.append(c)
        grp.sort(key=lambda c: ABL_LABEL[c["config_id"]].lower())
        rows.append(header(title))
        for c in grp:
            placed.add(c["config_id"])
            used.append(c)
            rows.append(row(c, ABL_LABEL[c["config_id"]],
                            "#000000" if c["family"] == "plain" else GREY))
    assert placed == {c["config_id"] for c in all_c}
    h = ROW_H * len(rows) + TOP_IN + BOTTOM_IN
    fig, ax = plt.subplots(figsize=(WIDTH, h), gridspec_kw=margins(h, 0.42))
    dot_panel(ax, rows, base=100 * cfg(E, "C_plain")["acc_mean"], xlim=(70.4, 82.2))
    ax.set_xlabel("final test accuracy after 30 epochs (%)")
    ax.set_title("ablation_aa (own initial weights)", loc="left")
    add_legend(fig, control=True)
    save(fig, "appendix_ablation.pdf", used,
         "ablation_aa, read at continuation-gaussian-tv_exploration_1@591e125; 1-seed arms "
         "have no SD; controls keep a fixed filter or BlurPool at inference")


def perlayer_label(c):
    lab = short_label(c["label"])
    lab = lab.replace("Baseline: no blur, full 32x32 throughout (per-layer batch)", "plain")
    lab = lab.replace("Per-layer blur strength", "per-layer \u03c3")
    return lab.replace(", [launch", " [launch").replace("sigma0", "\u03c3\u2080")


ADAPT_REPL = [(" (instrumented rerun)", ", instrumented"),
              ("Blur every conv layer, annealed", "conv-output plateaus"),
              ("Blur schedule chosen by a transfer-gap trigger", "\u03c3 schedule, transfer-gap trigger"),
              ("Blur schedule chosen by the network (first trigger)",
               "\u03c3 schedule chosen by the network"),
              ("Blur schedule, more time at", "\u03c3 schedule, more time at"),
              ("Resolution schedule chosen by a transfer-gap trigger",
               "resolution schedule, transfer-gap trigger")]


def adaptive_label(c):
    lab = short_label(c["label"])
    for a, b in ADAPT_REPL:
        lab = lab.replace(a, b)
    return lab


def fig_perlayer_adaptive():
    P, A = "per_layer_sigma", "adaptive_continuation"

    def full(c):
        return c["run_conditions"].get("epochs") == 30 and c["run_conditions"].get("n_train") == 50000
    pr = sorted([c for c in cfgs_of(P) if full(c)],
                key=lambda c: (c["family"] != "plain", c["label"].lower()))
    ar = sorted([c for c in cfgs_of(A) if full(c)], key=lambda c: c["label"].lower())
    rows_p = [row(c, perlayer_label(c), "#000000" if c["family"] == "plain" else GREY) for c in pr]
    rows_a = [row(c, adaptive_label(c)) for c in ar]
    fig, a, b = stacked(rows_p, rows_a, 0.45)
    dot_panel(a, rows_p, xlim=(69.0, 80.4))
    a.set_title("(a) per_layer_sigma, 30 epochs", loc="left")
    dot_panel(b, rows_a, xlim=(69.0, 80.4))
    b.set_title("(b) adaptive_continuation, 30 epochs, seed 0", loc="left")
    b.set_xlabel("final test accuracy after 30 epochs (%)")
    add_legend(fig, base=False)
    save(fig, "appendix_perlayer_adaptive.pdf", pr + ar,
         "per_layer_sigma and adaptive_continuation; CPU pilots, smoke runs and 120-epoch runs "
         "are excluded from the figure and listed in the complete table")


# --------------------------------------------------------------------------- main

def main():
    check_against_core()
    TAB.mkdir(parents=True, exist_ok=True)
    numbers()
    tables_unified()
    tables_losses()
    tables_operators()
    tables_plain_arms()
    tables_assets()
    tables_schedule()
    complete_table()
    fig_unified()
    fig_unified_curves()
    fig_resolution_operators()
    fig_resolution_location_schedule()
    fig_campaign()
    fig_ablation()
    fig_perlayer_adaptive()
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--",
                                 "continuation_core", "paper/tools"],
                                capture_output=True, text=True).stdout.strip())
    PROV.update({
        "generator": "paper/tools/make_paper_assets.py", "command": COMMAND,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manuscript_head_at_generation": head,
        "generator_or_core_uncommitted_at_generation": dirty,
        "benchmark": {"branch": BENCH_BRANCH, "commit": BENCH_COMMIT, "index": INDEX_PATH,
                      "index_built_from_commit": IX["built_from_commit"],
                      "pinned_refs": IX["pinned_refs"]},
        "software": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                     "numpy": np.__version__},
        "conventions": {
            "accuracy": "final record of each run (no best-epoch selection), percent",
            "bands_and_bars": "+/- 1 sample SD over valid seeds (ddof=1); never a confidence "
                              "interval; one-seed arms have none",
            "paired_differences": "same seed on the same asset set; SD of per-seed differences",
            "curves_x": "record k = evaluation after k completed epochs; current path uses "
                        "the state of zero-based epoch k-1; target = bypass32 evaluation"}})
    PROV_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROV_OUT.write_text(json.dumps(PROV, indent=1) + "\n", encoding="utf-8")
    print("wrote provenance/generated_assets.json")


if __name__ == "__main__":
    main()

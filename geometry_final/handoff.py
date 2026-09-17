"""LaTeX-ready tables, number macros and proposed text for the geometry section.

    py -m geometry_final.handoff

Writes ``studies/geometry_final/paper_handoff/``.  Nothing in the manuscript is edited.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import common as C

TAB = C.STUDY / "tables"
OUT = C.STUDY / "paper_handoff"
M = list(C.METHODS)
NAME = dict(zip(M, ["Plain", "R", "G", "RG"]))
LONGN = dict(zip(M, ["Plain", "Resolution (R)", "Gaussian (G)", "Combined (RG)"]))
CONTEXT = ("CIFAR-10 / ResNet-20-BN / SGD, final 30-epoch checkpoints, five training seeds; mean cross-entropy "
           "without weight decay on the 1k-image probe; centre-frozen BatchNorm (recalibrated once at the unperturbed "
           "weights on 2,000 training images, then held fixed); variables: the 268,336 convolution and classifier "
           "weights (698 filter/row blocks); ordinary coordinates unless marked $A^\\top HA$ with $A_g=\\|\\theta_g\\|I$.")


def pm(m, s, d=0):
    return "$%s\\pm%s$" % (("{:,.%df}" % d).format(m).replace(",", "\\,"), ("{:,.%df}" % d).format(s).replace(",", "\\,"))


def table(env, label, caption, cols, header, rows, foot=""):
    body = ("\\begin{%s}[!t]\n\\caption{%s}\\label{%s}\n\\centering\\small\\setlength{\\tabcolsep}{4pt}"
            "\\renewcommand{\\arraystretch}{1.08}\n\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}%s@{}}\n"
            "\\toprule\n%s\\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular*}\n" % (env, caption, label, cols, header, "\n".join(rows)))
    if foot:
        body += "\\par\\smallskip\\begin{minipage}{\\linewidth}\\footnotesize %s\\end{minipage}\n" % foot
    return body + "\\end{%s}\n" % env


def primary(ss, pc):
    rows = []
    q = [("lambda_top1_ordinary", 0), ("trace_ordinary", 0), ("lambda_top1_relative", 0), ("trace_relative", 0),
         ("trace_covariance", 1)]
    for probe in C.PROBES:
        rows.append("\\addlinespace[3pt]\\multicolumn{6}{@{}l@{}}{\\textit{%s probe (1,000 images)}}\\\\[1pt]"
                    % ("Training" if probe == "train_probe" else "Test"))
        for m in M:
            cells = []
            for col, d in q:
                r = ss[(ss.probe == probe) & (ss.method == m) & (ss.quantity == col)].iloc[0]
                cells.append(pm(r["mean"], r.sd_over_seeds, d))
            rows.append(LONGN[m] + " & " + " & ".join(cells) + "\\\\")
    tr = pc[[c for c in pc.columns if c.endswith("_mc_sem_rel")]]
    foot = ("Mean $\\pm$ sample SD across the five training seeds. $\\lambda_{\\max}$: Lanczos with full "
            "reorthogonalisation, two starts, explicit relative residual $\\le10^{-3}$ (all converged). Traces: Hutchinson "
            "estimates from 128 Rademacher draws per checkpoint, shared by the four methods within a seed; within-checkpoint "
            "Monte Carlo SEM is %.1f--%.1f\\%% of each estimate (not included in the $\\pm$). $\\mathrm{tr}(HC)$ is the "
            "expected quadratic form $\\mathbb E[d^\\top Hd]$ under our filter-normalised random directions "
            "($C_g=\\|\\theta_g\\|^2/p_g\\,I$). Traces are signed; all 40 objectives also have a negative eigenvalue."
            % (100 * tr.min().min(), 100 * tr.max().max()))
    return table("table*", "tab:geometry-primary",
                 "Curvature of the final solutions under one fixed objective. " + CONTEXT,
                 "lrrrrr", "Method & $\\lambda_{\\max}(H)$ & $\\mathrm{tr}(H)$ & $\\lambda_{\\max}(A^\\top HA)$ & "
                           "$\\mathrm{tr}(A^\\top HA)$ & $\\mathrm{tr}(HC)$", rows, foot)


def ratios(ss):
    q = ["lambda_top1_ordinary", "trace_ordinary", "lambda_top1_relative", "trace_relative", "trace_covariance", "quadform_20dir_mean"]
    rows = []
    for probe in C.PROBES:
        rows.append("\\addlinespace[3pt]\\multicolumn{7}{@{}l@{}}{\\textit{%s probe}}\\\\[1pt]" % ("Training" if probe == "train_probe" else "Test"))
        for m in M[1:]:
            cells = []
            for col in q:
                r = ss[(ss.probe == probe) & (ss.method == m) & (ss.quantity == col)].iloc[0]
                if col.startswith("trace"):
                    lab = "%d/%d" % (r.n_seeds_below_plain_resolved_2sem, r.n_seeds_above_plain_resolved_2sem)
                else:
                    lab = "%d" % r.n_seeds_below_plain
                cells.append("$%.2f\\pm%.2f$ {\\scriptsize(%s)}" % (r.ratio_to_plain_mean, r.ratio_to_plain_sd, lab))
            rows.append(NAME[m] + " & " + " & ".join(cells) + "\\\\")
    foot = ("Within-seed ratio to Plain, mean $\\pm$ SD over five seeds. In brackets: seeds below Plain for eigenvalues and the "
            "20-direction mean; for traces, seeds whose paired difference from Plain is below/above zero by more than two "
            "Monte Carlo SEM (paired draws); the remaining seeds are unresolved. The last column is the existing average of "
            "$d_k^\\top Hd_k$ over the 20 sampled directions, an independent Monte Carlo estimate of $\\mathrm{tr}(HC)$.")
    return table("table*", "tab:geometry-ratios", "Curvature relative to Plain within each seed. " + CONTEXT, "lrrrrrr",
                 "Method & $\\lambda_{\\max}(H)$ & $\\mathrm{tr}(H)$ & $\\lambda_{\\max}(A^\\top HA)$ & $\\mathrm{tr}(A^\\top HA)$ & "
                 "$\\mathrm{tr}(HC)$ & $\\overline{d^\\top Hd}$ (20)", rows, foot)


def bn_policy():
    s = pd.read_csv(TAB / "X1_bn_policy_finite_summary.csv")
    rows = []
    for split in C.PROBES:
        rows.append("\\addlinespace[3pt]\\multicolumn{7}{@{}l@{}}{\\textit{%s probe}}\\\\[1pt]" % ("Training" if split == "train_probe" else "Test"))
        for m in M:
            cells = []
            for eps in (0.01, 0.1, 0.5):
                r = s[(s.method == m) & (s.split == split) & (s.amplitude == eps)].iloc[0]
                cells.append("$%.2f$" % r.ratio_pw_over_cf_median)
            for eps in (0.01, 0.1, 0.5):
                r = s[(s.method == m) & (s.split == split) & (s.amplitude == eps)].iloc[0]
                cells.append("---" if m == "plain" else "$%+.0f$\\%% (%d) / $%+.0f$\\%% (%d)" % (
                    100 * r.relchange_centre_frozen_mean, r.n_below_plain_centre_frozen,
                    100 * r.relchange_pointwise_mean, r.n_below_plain_pointwise))
            rows.append(NAME[m] + " & " + " & ".join(cells) + "\\\\")
    x2 = pd.read_csv(TAB / "X2_bn_policy_eigencuts.csv")
    ec = []
    for t in (0.001, 0.01, 0.1):
        g = x2[(x2.t == t) & (x2.vector == "top1")]
        ec.append("$t=%g$: %.2g vs %.2g nats" % (t, g.train_probe_rise_centre_frozen.median(), g.train_probe_rise_pointwise.median()))
    foot = ("Same checkpoints, 20 directions, both signs and amplitudes; only the BatchNorm evaluation changes. Left: median "
            "over seeds of $S_{\\rm pointwise}/S_{\\rm centre\\mbox{-}frozen}$. Right: mean relative change of $S$ against Plain "
            "under centre-frozen / pointwise statistics, with the number of seeds below Plain. Along each model's leading "
            "block-relative eigendirection (training probe, 20 seed-method pairs, median rise): %s." % "; ".join(ec))
    return table("table*", "tab:geometry-bn-policy",
                 "Finite perturbations under centre-frozen and pointwise-recalibrated BatchNorm. CIFAR-10 / ResNet-20-BN / SGD, "
                 "final checkpoints, five seeds, 1k-image probes, filter-normalised directions.",
                 "lrrrccc", "Method & \\multicolumn{3}{c}{$S_{\\rm pw}/S_{\\rm cf}$ at $\\varepsilon=0.01,0.1,0.5$} & "
                            "$\\varepsilon=0.01$ & $\\varepsilon=0.1$ & $\\varepsilon=0.5$", rows, foot)


def amplitude():
    r = pd.read_csv(TAB / "X4_amplitude_random.csv")
    e = pd.read_csv(TAB / "X4_amplitude_eigen.csv")
    e = e[e.policy == "centre_frozen"]
    rows = []
    for probe in C.PROBES:
        for m in M:
            cells = []
            for eps in (0.005, 0.05, 0.2, 0.5):
                x = r[(r.policy == "centre_frozen") & (r.probe == probe) & (r.method == m) & (r.amplitude == eps)].iloc[0]
                cells.append("$%.2f$" % x["median"])
            rows.append("%s, %s & %s\\\\" % (NAME[m], "train" if probe == "train_probe" else "test", " & ".join(cells)))
    ev = []
    for vec in ("top1", "min"):
        for t in (1e-5, 1e-4, 1e-3, 1e-2):
            g = e[(e.vector == vec) & (np.isclose(e.t, t))]
            ev.append("%s, $t=10^{%d}$: median %.2f, sign agrees %d/%d" % (vec, int(np.log10(t)), g.ratio_C_fd_over_quad.median(),
                                                                         g.sign_agrees.sum(), len(g)))
    foot = ("Median over five seeds of $C(\\varepsilon)/\\overline{d^\\top Hd}$, where $C=2S/\\varepsilon^2$ averages the 20 "
            "random directions and $\\overline{d^\\top Hd}$ is their HVP quadratic form (same objective). Along existing "
            "eigendirections scaled to $r=1$ (float64 finite differences, 80 centre-frozen problems): %s." % "; ".join(ev))
    return table("table", "tab:geometry-amplitude", "Local quadratic forms against measured finite differences, centre-frozen BatchNorm. "
                 "CIFAR-10 / ResNet-20-BN / SGD final checkpoints, five seeds, 1k-image probes.",
                 "lrrrr", "Method, probe & $\\varepsilon=0.005$ & $0.05$ & $0.2$ & $0.5$", rows, foot)


def studies():
    rows = [
        "Checkpoints & landscape\\_v2 runs, 5 seeds (sha256 pinned) & Idriss grid reference cells, 3 seeds & same as grid\\\\",
        "Parameters & 268,336 conv + fc weights & all 269,722 learnable & all 269,722 learnable\\\\",
        "Images & 1k class-balanced train / test probe & 5k training subset & first 2k train or test, batches of 500\\\\",
        "BatchNorm & centre-frozen (saved as check) & saved statistics & saved, or train-mode batch statistics\\\\",
        "Estimator & $\\lambda$: Lanczos; traces: 128 draws & 64 draws, 5-pair deflation & 24 draws, 1-pair deflation\\\\",
        "Coordinates & ordinary, block-relative, $C$-weighted & ordinary & ordinary\\\\",
    ]
    foot = ("The train-mode column differentiates through each batch's own statistics; it is neither the centre-frozen objective "
            "nor pointwise recalibration of running statistics. The studies agree in direction (all three procedures have smaller "
            "ordinary traces than Plain) but are not pooled numerically.")
    return table("table*", "tab:geometry-studies", "Curvature measurements in the two studies and their conventions.", "llll",
                 " & Landscape cohort (this section) & Endpoint grid & BN-policy check", rows, foot)


def macros(ss, pc):
    out = ["% GENERATED by geometry_final/handoff.py -- do not edit by hand"]

    def mac(name, val):
        out.append("\\newcommand{\\%s}{%s}" % (name, val))
    for probe, P in (("train_probe", "Train"), ("test_probe", "Test")):
        for m in M[1:]:
            for q, Q in (("lambda_top1_ordinary", "Lam"), ("trace_ordinary", "Tr"), ("trace_relative", "TrRel"), ("trace_covariance", "TrCov")):
                r = ss[(ss.probe == probe) & (ss.method == m) & (ss.quantity == q)].iloc[0]
                mac("geo%s%s%sRatio" % (NAME[m], P, Q), "%.2f" % r.ratio_to_plain_mean)
                if q.startswith("trace"):
                    mac("geo%s%s%sBelow" % (NAME[m], P, Q), "%d" % r.n_seeds_below_plain_resolved_2sem)
    mac("geoTraceDraws", "128")
    mac("geoTraceSemMaxPct", "%.1f" % (100 * pc[[c for c in pc.columns if c.endswith("_mc_sem_rel")]].max().max()))
    ext = (pc.r_units_extreme_top1 / pc.trace_covariance)
    mac("geoExtremeOverRandomMin", "{:,.0f}".format(ext.min()).replace(",", "\\,"))
    mac("geoExtremeOverRandomMax", "{:,.0f}".format(ext.max()).replace(",", "\\,"))
    return "\n".join(out) + "\n"


INSERT = r"""% Proposed insertions for sec:visualization (not applied). Numbers come from numbers_geometry.tex.

% (1) After the Hessian paragraph:
Leading eigenvalues describe one direction. We therefore also estimate traces of the same centre-frozen objective from
\geoTraceDraws{} Rademacher probes per checkpoint, shared across methods within a seed (Monte Carlo SEM at most
\geoTraceSemMaxPct\% of each estimate; Table~\ref{tab:geometry-primary}). The ordinary trace is smaller than Plain for R, G and RG
in every seed on both probes (training-probe ratios \geoRTrainTrRatio, \geoGTrainTrRatio{} and \geoRGTrainTrRatio), as is the
block-relative trace. Smaller leading eigenvalues therefore accompany smaller traces in these models and conventions.

% (2) Why random slices and eigenvalues rank G differently:
Our random directions do not sample curvature isotropically. Their covariance weights each filter by
$\|\theta_g\|^2/p_g$, so their mean curvature is $\mathrm{tr}(HC)$ rather than $\mathrm{tr}(H)$ or $\lambda_{\max}$. For G this
quantity is close to Plain on the training probe (ratio \geoGTrainTrCovRatio; \geoGTrainTrCovBelow{} of five seeds resolved below
Plain), although its leading eigenvalue is about half of Plain's. R and RG are lower on every measure. The steepest direction
is \geoExtremeOverRandomMin--\geoExtremeOverRandomMax{} times steeper per unit relative displacement than the average random
direction, so a two-dimensional random slice cannot display it (proposed Figure~\ref{fig:hessian-vs-random}).

% (3) BatchNorm policy:
Recalibrating at every perturbed point changes the function, not only the scale. For identical perturbations of Plain it
reduces $S$ by factors of about 3 to 8 (more at larger amplitude), at most amplitudes it reverses the sign of G's difference from Plain, and
along the leading eigendirections it removes almost all of the rise (Table~\ref{tab:geometry-bn-policy}). Those
eigendirections are not predominantly filter rescalings (at most 1\% of their squared relative displacement is radial), so the
reduction is not explained by the rescaling invariance of BatchNorm alone.

% (4) Scope sentence:
None of these quantities orders the four procedures in the same way as their test accuracy, and none is shown to cause it.
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ss = pd.read_csv(TAB / "T_trace_seed_summary.csv")
    pc = pd.read_csv(TAB / "T_trace_per_checkpoint.csv")
    (OUT / "tab_geometry_primary.tex").write_text(primary(ss, pc), encoding="utf-8")
    (OUT / "tab_geometry_ratios.tex").write_text(ratios(ss), encoding="utf-8")
    (OUT / "tab_geometry_bn_policy.tex").write_text(bn_policy(), encoding="utf-8")
    (OUT / "tab_geometry_amplitude.tex").write_text(amplitude(), encoding="utf-8")
    (OUT / "tab_geometry_studies.tex").write_text(studies(), encoding="utf-8")
    (OUT / "numbers_geometry.tex").write_text(macros(ss, pc), encoding="utf-8")
    (OUT / "proposed_insertions.tex").write_text(INSERT, encoding="utf-8")
    print("written", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()

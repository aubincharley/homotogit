"""Collect, check and analyse the 42-cell comparison; tables (CSV + LaTeX) and vector figures.

    py -m comparison.analyze

Reads ``studies/comparison_cbs_sdpoint/raw/main/<account>/cmp/cells/<run>/``.  Writes
``results/`` (tidy CSV), ``figures/`` (PDF + PNG + CAPTIONS.json) and ``paper/`` (LaTeX).
No exclusions: every finite run is kept.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import matrix as MX  # noqa: E402

RAW = MX.STUDY / "raw" / "main"
RES = MX.STUDY / "results"
FIG = MX.STUDY / "figures"
PAPER = MX.STUDY / "paper"
ARMS = list(MX.ARMS)
LABEL = {"plain": "Plain", "resolution_max_b1": "R", "gaussian_postrelu": "G", "resolution_max_b1_gaussian_conv": "RG",
         "cbs_published_schedule": "CBS (published)", "cbs_budget_matched": "CBS (budget-matched)", "sdpoint": "SDPoint"}
COLOR = {"plain": "#333333", "resolution_max_b1": "#1976b2", "gaussian_postrelu": "#dc7b13",
         "resolution_max_b1_gaussian_conv": "#8653aa", "cbs_published_schedule": "#2a9d8f",
         "cbs_budget_matched": "#7fc8bd", "sdpoint": "#c0392b"}
SETTING = {"sgd": "SGD (lr 0.005)", "adamw": "AdamW (lr 0.02)"}
COMPARISONS = [(a, "plain") for a in ARMS[1:]] + [
    ("gaussian_postrelu", "cbs_published_schedule"), ("gaussian_postrelu", "cbs_budget_matched"),
    ("resolution_max_b1", "sdpoint"), ("resolution_max_b1_gaussian_conv", "resolution_max_b1"),
    ("cbs_budget_matched", "cbs_published_schedule")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
CAPTIONS = {}


def load_cells():
    rows, problems = [], []
    for acc_dir in sorted(RAW.iterdir()) if RAW.exists() else []:
        for cell in sorted((acc_dir / "cmp" / "cells").glob("*")) if (acc_dir / "cmp" / "cells").exists() else []:
            setting, arm, seed = cell.name.split("__")
            seed = int(seed.replace("seed", ""))
            if not (cell / "DONE.json").exists():
                problems.append({"run": cell.name, "account": acc_dir.name,
                                 "status": "failed" if (cell / "FAILED.json").exists() else "incomplete"})
                continue
            p = json.loads((cell / "panels.json").read_text())
            t = json.loads((cell / "timing.json").read_text())
            s = json.loads((cell / "summary.json").read_text())
            action = [c for c in MX.cells() if c["run"] == cell.name][0]["action"]
            base = {"setting": setting, "arm": arm, "seed": seed, "run": cell.name, "account": acc_dir.name, "action": action,
                    "checkpoint_sha256": p["checkpoint_sha256"], "final_updates": p["global_update"],
                    "recorded_check_matches": p["recorded_check"]["matches"],
                    "recorded_check_abs_diff_acc": p["recorded_check"]["abs_diff_acc"],
                    "repeat_evaluation_identical": p["repeat_saved_native_identical"],
                    "native_equals_original": p["native_equals_original"],
                    "native_state": json.dumps(p["saved_native"]["state"]),
                    "native_per_site_sigma_max": max(p["saved_native"]["per_site_sigma"] or [0.0])}
            for key, panel in (("panelA", "panelA_recalibrated_native"), ("panelB", "panelB_recalibrated_original"),
                               ("saved_native", "saved_native"), ("saved_original", "saved_original")):
                rec = p[panel]
                if "same_as" in rec:
                    rec = p[rec["same_as"]]
                for split in ("test_full", "train_full"):
                    base["%s_%s_acc" % (key, split)] = rec[split]["acc"]
                    base["%s_%s_ce" % (key, split)] = rec[split]["ce"]
            tr = t.get("training", {})
            if action == "reuse_landscape_v2_checkpoint":
                rt = tr.get("recorded_timing") or {}
                base.update({"train_loop_seconds": rt.get("train_seconds"), "epoch_eval_seconds": rt.get("eval_seconds"),
                             "run_wall_seconds": rt.get("wall_seconds"),
                             "timing_scope": "landscape_v2 run: includes per-epoch and transition-window checkpoint writes; other session"})
            else:
                base.update({"train_loop_seconds": tr.get("train_seconds"), "epoch_eval_seconds": tr.get("eval_seconds"),
                             "run_wall_seconds": tr.get("wall_seconds"), "peak_cuda_memory_mib": tr.get("peak_cuda_memory_mib"),
                             "timing_scope": "this study: Trainer.run wall (train loop incl. final/rolling checkpoint writes + 31 epoch evaluations)"})
            base.update({"final_calibration_evaluation_seconds": t.get("final_calibration_and_evaluation_seconds"),
                         "cell_wall_seconds": t.get("cell_wall_seconds"), "gpu": t.get("gpu"),
                         "online_train_loss_last_epoch": s.get("final") and json.loads((cell / "metrics.json").read_text())[-1].get("train_loss_epoch"),
                         "recorded_final_test_acc_saved_bn": s["final"]["target" if arm in MX.HISTORICAL else "current"]["test"]["acc"]})
            rows.append(base)
    return pd.DataFrame(rows), problems


def curves():
    out = []
    for acc_dir in sorted(RAW.iterdir()) if RAW.exists() else []:
        for m in sorted((acc_dir / "cmp" / "cells").glob("*/metrics.json")):
            setting, arm, seed = m.parent.name.split("__")
            for rec in json.loads(m.read_text()):
                for path, d in rec["eval"].items():
                    out.append({"setting": setting, "arm": arm, "seed": int(seed[4:]), "epoch": rec["epoch"], "path": path,
                                "train_loss_epoch_online": rec.get("train_loss_epoch"),
                                "probe500_train_acc_saved_bn": d["train_probe"]["acc"], "probe500_train_ce_saved_bn": d["train_probe"]["ce"],
                                "test_acc_saved_bn": d["test"]["acc"], "test_ce_saved_bn": d["test"]["ce"]})
    return pd.DataFrame(out)


def summarise(df):
    rows, pairs = [], []
    for setting in MX.SETTINGS:
        for panel in ("panelA", "panelB", "saved_native", "saved_original"):
            for arm in ARMS:
                g = df[(df.setting == setting) & (df.arm == arm)].sort_values("seed")
                for split in ("test_full", "train_full"):
                    acc, ce = 100 * g["%s_%s_acc" % (panel, split)], g["%s_%s_ce" % (panel, split)]
                    rows.append({"setting": setting, "panel": panel, "arm": arm, "split": split, "n_seeds": len(g),
                                 "acc_mean_pct": acc.mean(), "acc_sd_pct": acc.std(ddof=1), "ce_mean": ce.mean(), "ce_sd": ce.std(ddof=1),
                                 **{"acc_seed%d_pct" % s: v for s, v in zip(g.seed, acc)}, **{"ce_seed%d" % s: v for s, v in zip(g.seed, ce)}})
            for a, b in COMPARISONS:
                ga = df[(df.setting == setting) & (df.arm == a)].set_index("seed")
                gb = df[(df.setting == setting) & (df.arm == b)].set_index("seed")
                seeds = sorted(set(ga.index) & set(gb.index))
                for split in ("test_full", "train_full"):
                    da = 100 * (ga.loc[seeds, "%s_%s_acc" % (panel, split)] - gb.loc[seeds, "%s_%s_acc" % (panel, split)])
                    dc = ga.loc[seeds, "%s_%s_ce" % (panel, split)] - gb.loc[seeds, "%s_%s_ce" % (panel, split)]
                    pairs.append({"setting": setting, "panel": panel, "comparison": "%s - %s" % (a, b), "split": split, "n_pairs": len(seeds),
                                  "acc_diff_mean_pp": da.mean(), "acc_diff_sd_pp": da.std(ddof=1), "n_positive": int((da > 0).sum()),
                                  "n_negative": int((da < 0).sum()), "ce_diff_mean": dc.mean(), "ce_diff_sd": dc.std(ddof=1),
                                  **{"acc_diff_seed%d_pp" % s: v for s, v in zip(seeds, da)}})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def save(fig, name, caption, sources):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(FIG / (name + ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    CAPTIONS[name] = {"caption": caption, "sources": sources}


def fig_accuracy(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    for ax, setting in zip(axes, MX.SETTINGS):
        for i, arm in enumerate(ARMS):
            g = df[(df.setting == setting) & (df.arm == arm)]
            y = 100 * g.panelA_test_full_acc
            ax.plot(np.full(len(y), i) + np.linspace(-.12, .12, len(y)), y, "o", ms=3.5, color=COLOR[arm])
            ax.plot([i - .3, i + .3], [y.mean()] * 2, color=COLOR[arm], lw=2)
            if arm == "cbs_published_schedule":
                yb = 100 * g.panelB_test_full_acc
                ax.plot(np.full(len(yb), i + .38), yb, "x", ms=3.5, color=COLOR[arm])
        ax.set_xticks(range(len(ARMS)), [LABEL[a] for a in ARMS], rotation=35, ha="right", fontsize=7.5)
        ax.set_title(SETTING[setting], fontsize=9)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Test accuracy (%)")
    cap = ("Final test accuracy of seven training procedures: CIFAR-10 (50,000 training / 10,000 test images), ResNet-20 with "
           "BatchNorm, no augmentation, 30 epochs, left SGD and right AdamW; three seeds (dots; bar = mean). Primary evaluation "
           "(Panel A): each procedure's native inference network, BatchNorm statistics recalibrated on all training images with "
           "the same protocol. Crosses: published-schedule CBS evaluated without its final Gaussian filters (Panel B). Vertical "
           "scales differ between panels.")
    save(fig, "C1_accuracy_panelA", cap, ["results/per_run.csv"])


def fig_cbs(df):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), layout="constrained")
    for ax, setting in zip(axes, MX.SETTINGS):
        g = df[(df.setting == setting)]
        cols = [("panelA_test_full_acc", "native,\nrecalibrated"), ("panelB_test_full_acc", "filters removed,\nrecalibrated"),
                ("saved_native_test_full_acc", "native,\nsaved BN"), ("saved_original_test_full_acc", "filters removed,\nsaved BN")]
        for arm, dx in (("cbs_published_schedule", -.12), ("cbs_budget_matched", .12), ("gaussian_postrelu", .0)):
            h = g[g.arm == arm]
            for j, (c, _) in enumerate(cols):
                y = 100 * h[c]
                ax.plot(np.full(len(y), j + dx), y, "o", ms=3.2, color=COLOR[arm], label=LABEL[arm] if j == 0 else None)
        ax.set_xticks(range(4), [c[1] for c in cols], fontsize=7)
        ax.set_title(SETTING[setting], fontsize=9)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[1].legend(fontsize=7, frameon=False, loc="lower left")
    cap = ("CBS endpoints with and without their final filters: CIFAR-10 / ResNet-20-BN, 30 epochs, three seeds per procedure. "
           "Published-schedule CBS ends with sigma = 0.59 filters in place; budget-matched CBS and G end unfiltered, so their "
           "two states coincide. Recalibrated: BatchNorm statistics re-estimated on all training images in the evaluated state; "
           "saved BN: the checkpoint's running statistics.")
    save(fig, "C2_cbs_native_vs_unfiltered", cap, ["results/per_run.csv"])


def fig_costs(df):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), layout="constrained", sharey=True)
    for ax, setting in zip(axes, MX.SETTINGS):
        for i, arm in enumerate(ARMS):
            g = df[(df.setting == setting) & (df.arm == arm) & (df.action != "reuse_landscape_v2_checkpoint")]
            if len(g) == 0:
                continue
            y = g.train_loop_seconds / 60
            ax.plot(np.full(len(y), i) + np.linspace(-.1, .1, len(y)), y, "o", ms=3.2, color=COLOR[arm])
        ax.set_xticks(range(len(ARMS)), [LABEL[a] for a in ARMS], rotation=35, ha="right", fontsize=7.5)
        ax.set_title(SETTING[setting], fontsize=9)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Training time (min)")
    cap = ("Measured training time of the runs trained in this comparison: 30 epochs of CIFAR-10 on one Tesla T4, float32, "
           "one run per GPU at a time, including 31 per-epoch evaluations and checkpoint writes. SGD Plain/R/G/RG were not "
           "retrained (their existing runs used another session and wrote more checkpoints), so they are omitted rather than "
           "compared.")
    save(fig, "C3_training_time", cap, ["results/per_run.csv"])


def tex_tables(summary, pairs, df):
    PAPER.mkdir(parents=True, exist_ok=True)

    def cell(setting, panel, arm, split="test_full"):
        r = summary[(summary.setting == setting) & (summary.panel == panel) & (summary.arm == arm) & (summary.split == split)].iloc[0]
        return "$%.2f\\pm%.2f$" % (r.acc_mean_pct, r.acc_sd_pct)

    def dcell(setting, panel, a, b="plain"):
        if a == b:
            return "---"
        r = pairs[(pairs.setting == setting) & (pairs.panel == panel) & (pairs.comparison == "%s - %s" % (a, b)) & (pairs.split == "test_full")].iloc[0]
        return "$%+.2f\\pm%.2f$" % (r.acc_diff_mean_pp, r.acc_diff_sd_pp)

    rows = []
    for arm in ARMS:
        rows.append("%s & %s & %s & %s & %s\\\\" % (LABEL[arm], cell("sgd", "panelA", arm), dcell("sgd", "panelA", arm),
                                                  cell("adamw", "panelA", arm), dcell("adamw", "panelA", arm)))
    t1 = ("\\begin{table*}[!t]\n\\caption{Comparison with Curriculum By Smoothing and SDPoint: final test accuracy (\\%%) on CIFAR-10 "
          "(50\\,000 training / 10\\,000 test images), ResNet-20 with BatchNorm, no augmentation, 30 epochs, three seeds. Native "
          "inference network of each procedure; BatchNorm statistics recalibrated on all training images with one protocol for all "
          "procedures. Mean $\\pm$ sample SD across seeds; $\\Delta$: seed-paired difference from Plain (percentage points), mean "
          "$\\pm$ SD of the three differences.}\\label{tab:comparators}\n\\centering\\small\\setlength{\\tabcolsep}{4pt}"
          "\\renewcommand{\\arraystretch}{1.08}\n\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrrr@{}}\n\\toprule\n"
          "& \\multicolumn{2}{c}{SGD, lr 0.005} & \\multicolumn{2}{c}{AdamW, lr 0.02}\\\\\n\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
          "Procedure & Accuracy & $\\Delta$ vs Plain & Accuracy & $\\Delta$ vs Plain\\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular*}\n"
          "\\par\\smallskip\\begin{minipage}{\\linewidth}\\footnotesize CBS (published): $\\sigma=0.9^{\\lfloor e/5\\rfloor}$, evaluated "
          "with its final filters. CBS (budget-matched): 22 update-level plateaus over the first 70\\%% of training, then no filter; "
          "our adaptation, not an author schedule. SDPoint: full-resolution instance; its reduced-cost instances are not evaluated. "
          "R, G and RG are our selected procedures; G is a CBS-inspired variant.\\end{minipage}\n\\end{table*}\n") % "\n".join(rows)
    (PAPER / "tab_comparators_panelA.tex").write_text(t1, encoding="utf-8")

    rows = []
    for setting in MX.SETTINGS:
        rows.append("\\addlinespace[3pt]\\multicolumn{5}{@{}l@{}}{\\textit{%s}}\\\\[1pt]" % SETTING[setting])
        for a, b in COMPARISONS[len(ARMS) - 1:]:
            ra = pairs[(pairs.setting == setting) & (pairs.panel == "panelA") & (pairs.comparison == "%s - %s" % (a, b)) & (pairs.split == "test_full")].iloc[0]
            rb = pairs[(pairs.setting == setting) & (pairs.panel == "panelB") & (pairs.comparison == "%s - %s" % (a, b)) & (pairs.split == "test_full")].iloc[0]
            rows.append("%s $-$ %s & $%+.2f\\pm%.2f$ & %d/%d & $%+.2f\\pm%.2f$ & $%+.3f\\pm%.3f$\\\\" % (
                LABEL[a], LABEL[b], ra.acc_diff_mean_pp, ra.acc_diff_sd_pp, ra.n_positive, ra.n_pairs,
                rb.acc_diff_mean_pp, rb.acc_diff_sd_pp, ra.ce_diff_mean, ra.ce_diff_sd))
    t2 = ("\\begin{table}[!t]\n\\caption{Predeclared paired comparisons: CIFAR-10 / ResNet-20-BN, 30 epochs, three seeds, full test set. "
          "Accuracy differences in percentage points and test cross-entropy differences in nats, mean $\\pm$ SD of the three "
          "seed-paired differences; A: native network, B: added interventions removed; both with recalibrated BatchNorm.}"
          "\\label{tab:comparators-paired}\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.08}\n"
          "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrrr@{}}\n\\toprule\nComparison & $\\Delta$acc A & $>0$ & $\\Delta$acc B & $\\Delta$CE A\\\\\n"
          "\\midrule\n%s\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n") % "\n".join(rows)
    (PAPER / "tab_comparators_paired.tex").write_text(t2, encoding="utf-8")

    rows = []
    for setting in MX.SETTINGS:
        rows.append("\\addlinespace[3pt]\\multicolumn{7}{@{}l@{}}{\\textit{%s}}\\\\[1pt]" % SETTING[setting])
        for arm in ARMS:
            vals = []
            for panel in ("panelA", "panelB", "saved_native", "saved_original"):
                vals.append(cell(setting, panel, arm))
            r = summary[(summary.setting == setting) & (summary.panel == "panelA") & (summary.arm == arm) & (summary.split == "train_full")].iloc[0]
            rc = summary[(summary.setting == setting) & (summary.panel == "panelA") & (summary.arm == arm) & (summary.split == "test_full")].iloc[0]
            rows.append("%s & %s & $%.2f\\pm%.2f$ & $%.3f$ / $%.3f$\\\\" % (LABEL[arm], " & ".join(vals), r.acc_mean_pct, r.acc_sd_pct, r.ce_mean, rc.ce_mean))
    t3 = ("\\begin{table*}[!t]\n\\caption{All evaluation policies: CIFAR-10 / ResNet-20-BN, 30 epochs, three seeds, mean $\\pm$ SD. "
          "Test accuracy (\\%%) of the native network with recalibrated BatchNorm (A), without added interventions and recalibrated (B), "
          "native with the checkpoint's saved statistics, and without interventions with saved statistics; training-set accuracy "
          "(50\\,000 images) and training/test cross-entropy under A. Saved-statistics values are diagnostics; for SDPoint they mix "
          "statistics over its training instances.}\\label{tab:comparators-policies}\n\\centering\\small\\setlength{\\tabcolsep}{3pt}"
          "\\renewcommand{\\arraystretch}{1.08}\n\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}\n\\toprule\n"
          "Procedure & Test A & Test B & Test saved (native) & Test saved (removed) & Train A & CE train / test A\\\\\n\\midrule\n%s\n"
          "\\bottomrule\n\\end{tabular*}\n\\end{table*}\n") % "\n".join(rows)
    (PAPER / "tab_comparators_policies.tex").write_text(t3, encoding="utf-8")

    rows = []
    for setting in MX.SETTINGS:
        rows.append("\\addlinespace[3pt]\\multicolumn{4}{@{}l@{}}{\\textit{%s}}\\\\[1pt]" % SETTING[setting])
        for arm in ARMS:
            g = df[(df.setting == setting) & (df.arm == arm)]
            if (g.action == "reuse_landscape_v2_checkpoint").all():
                rows.append("%s & \\multicolumn{3}{c}{not retrained (different session and checkpoint cadence)}\\\\" % LABEL[arm])
                continue
            rows.append("%s & $%.1f\\pm%.1f$ & $%.1f\\pm%.1f$ & $%.2f\\pm%.2f$\\\\" % (
                LABEL[arm], (g.train_loop_seconds / 60).mean(), (g.train_loop_seconds / 60).std(ddof=1),
                (g.epoch_eval_seconds / 60).mean(), (g.epoch_eval_seconds / 60).std(ddof=1),
                (g.final_calibration_evaluation_seconds / 60).mean(), (g.final_calibration_evaluation_seconds / 60).std(ddof=1)))
    t4 = ("\\begin{table}[!t]\n\\caption{Measured costs of the runs trained in this comparison (minutes, mean $\\pm$ SD over three seeds): "
          "one Tesla T4, float32, one run per GPU. Training: wall time of the training loop excluding per-epoch evaluation, including "
          "checkpoint writes; epoch evaluations: 31 evaluations of a 500-image training probe and the test set; final: all "
          "recalibrations and full-set evaluations of the endpoint.}\\label{tab:comparators-costs}\n\\centering\\small"
          "\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.08}\n\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrr@{}}\n"
          "\\toprule\nProcedure & Training & Epoch evaluations & Final evaluation\\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n") % "\n".join(rows)
    (PAPER / "tab_comparators_costs.tex").write_text(t4, encoding="utf-8")


def main():
    for d in (RES, FIG, PAPER):
        d.mkdir(parents=True, exist_ok=True)
    df, problems = load_cells()
    (RES / "problems.json").write_text(json.dumps(problems, indent=1))
    df = df.sort_values(["setting", "arm", "seed"], key=lambda c: c.map({a: i for i, a in enumerate(ARMS)}) if c.name == "arm" else c)
    df.to_csv(RES / "per_run.csv", index=False)
    curves().to_csv(RES / "training_curves.csv", index=False)
    summary, pairs = summarise(df)
    summary.to_csv(RES / "summary_by_arm.csv", index=False)
    pairs.to_csv(RES / "paired_comparisons.csv", index=False)
    fig_accuracy(df)
    fig_cbs(df)
    fig_costs(df)
    tex_tables(summary, pairs, df)
    (FIG / "CAPTIONS.json").write_text(json.dumps(CAPTIONS, indent=1, ensure_ascii=False))
    checks = {"n_runs_complete": len(df), "expected": 42, "problems": problems,
              "all_recorded_checks_match": bool(df.recorded_check_matches.all()),
              "all_repeat_evaluations_identical": bool(df.repeat_evaluation_identical.all()),
              "all_final_updates_11730": bool((df.final_updates == MX.TOTAL_UPDATES).all()),
              "distinct_checkpoints": int(df.checkpoint_sha256.nunique())}
    (RES / "collection_checks.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()

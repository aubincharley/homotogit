"""Offline analysis of the adaptive-resolution Phase 0 job.

Reads ``results/kaggle_outputs/<slug>/adaptive_phase0_*/*/metrics.json`` and
answers, with numbers only from those traces:

1. replication -- final accuracies of the six Part-A cells against the campaign;
2. the plan's go / no-go rule (section 11.4): relative decrease of the smoothed
   probe gradient norm over the last two epochs of each stage;
3. post-switch spike length and amplitude, which sets the blackout ``B``;
4. rank correlation between ||g|| and monitor CE within a stage (kill criterion);
5. trigger replay -- first epoch at which each candidate criterion would fire,
   per seed, against the fixed boundaries;
6. the schedule grid ranking (Part B, seed 0, exploratory);
7. Q-04 -- how much of the target-path collapse BatchNorm recalibration recovers.

No training is run; nothing is written outside ``results/``.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CAMPAIGN = {  # final test accuracy of the paired campaign cells (EXP-011)
    "R32__Gnone__input_bilinear__seed0": 0.7513, "R32__Gnone__input_bilinear__seed1": 0.7562,
    "R32__Gnone__input_bilinear__seed2": 0.7610,
    "Rprog__Gnone__input_bilinear__seed0": 0.7920, "Rprog__Gnone__input_bilinear__seed1": 0.7910,
    "Rprog__Gnone__input_bilinear__seed2": 0.8010,
    "Rprog__Gplateau__input_bilinear__seed0": 0.8015, "Rprog__Gplateau__input_bilinear__seed1": 0.8057,
    "Rprog__Gplateau__input_bilinear__seed2": 0.8142,
}


def load(slug_glob):
    runs = {}
    for p in glob.glob(str(ROOT / "results" / "kaggle_outputs" / slug_glob
                           / "adaptive_phase*_*" / "*" / "metrics.json")):
        label = Path(p).parent.name
        if (Path(p).parent / "INVALID.json").is_file():      # runs invalidated after the fact
            continue
        m = json.load(open(p))
        s = Path(p).parent / "summary.json"
        runs[label] = {"metrics": m, "summary": json.load(open(s)) if s.is_file() else None}
    return runs


def series(m, key_fn):
    return np.array([key_fn(r) for r in m], dtype=float)


def role(r, name):
    """Signal entry by role ('current' / 'next' / 'target'), for both trace formats."""
    s = r["signals"]
    if "current" in s:
        return s.get(name, {})
    if name == "current":
        return s.get(str(r["resolution_current"]), {})
    if name == "next":
        return s.get(str(r["resolution_next"]), {}) if r["resolution_next"] is not None else {}
    return s.get("32", {}) if r["resolution_current"] != 32 else {}


def cur_sig(r, k):
    return role(r, "current")[k]


def ema(x, beta=0.5):
    out, v = [], None
    for xi in x:
        v = xi if v is None else beta * v + (1 - beta) * xi
        out.append(v)
    return np.array(out)


def stages(res):
    """[(start_epoch_index, end_epoch_index_exclusive, resolution)] over the record list."""
    out, start = [], 0
    for i in range(1, len(res) + 1):
        if i == len(res) or res[i] != res[start]:
            out.append((start, i, res[start]))
            start = i
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def relative_decrease_trigger(x, eps, window=2, patience=2, blackout=1):
    """First index at which the relative decrease over ``window`` checks stays below eps
    for ``patience`` consecutive checks; ``blackout`` leading checks are discarded."""
    x = x[blackout:]
    hits, fired = 0, None
    for k in range(window, len(x)):
        d = (x[k - window] - x[k]) / max(abs(x[k - window]), 1e-12)
        hits = hits + 1 if d < eps else 0
        if hits >= patience:
            fired = k + blackout
            break
    return fired


def analyse(runs, out_json):
    report = {"replication": {}, "go_no_go": {}, "spikes": {}, "rank_corr": {},
              "trigger_replay": {}, "grid": {}, "q04_bn_share": {}}
    lines = []

    # 1. replication
    lines.append("## 1. Replication of the six paired campaign cells\n")
    lines.append("| cell | this job | campaign | diff (pp) |\n|---|---:|---:|---:|")
    for label, ref in CAMPAIGN.items():
        if label in runs and runs[label]["summary"]:
            acc = runs[label]["summary"]["final_test_acc"]
            report["replication"][label] = {"job": acc, "campaign": ref, "diff_pp": 100 * (acc - ref)}
            lines.append("| %s | %.4f | %.4f | %+.2f |" % (label, acc, ref, 100 * (acc - ref)))

    # 2-5. per-run stage analysis
    for label, run in sorted(runs.items()):
        m = run["metrics"]
        if not m or m[-1]["epoch"] < 2:
            continue
        res = [r["resolution_current"] for r in m]
        g = series(m, lambda r: cur_sig(r, "grad_norm")["total"])
        gt = series(m, lambda r: role(r, "current").get("grad_norm_trainbn", {}).get("total", float("nan")))
        ce = series(m, lambda r: cur_sig(r, "monitor_ce"))
        gs = ema(g)
        st = stages(res)
        gn, sp, rc, rep = {}, {}, {}, {}
        for si, (a, b, r) in enumerate(st):
            if b - a >= 3 and si < len(st) - 1:          # a stage that ends in a switch
                D = (gs[b - 3] - gs[b - 1]) / max(gs[b - 3], 1e-12)   # last two epochs
                gn["stage%d_r%d" % (si, r)] = {"D_last2": float(D), "epochs": [m[a]["epoch"], m[b - 1]["epoch"]],
                                               "verdict": "GO" if D < 0.05 else ("NO-GO" if D > 0.10 else "marginal")}
            if b - a >= 4:
                rc["stage%d_r%d" % (si, r)] = {"spearman_g_vs_ce": spearman(g[a:b], ce[a:b]),
                                               "n": int(b - a)}
            if si > 0:                                     # spike after entering this stage
                pre = g[a - 1]
                post = g[a:min(a + 4, b)]
                amp = float(post[0] / pre) if pre > 0 else float("nan")
                # spike length: checks until g falls back below the pre-switch level
                length = 0
                for v in post:
                    if v > pre:
                        length += 1
                    else:
                        break
                sp["enter_r%d_at_epoch%d" % (r, m[a]["epoch"])] = {"amplitude_ratio": amp, "length_checks": length,
                                                                     "pre": float(pre), "post": [float(v) for v in post]}
            # trigger replay within this stage (only for stages followed by a switch)
            if si < len(st) - 1 and b - a >= 3:
                seg = slice(a, b)
                fires = {}
                for eps in (0.02, 0.05, 0.10, 0.20):
                    f = relative_decrease_trigger(gs[seg], eps)
                    fires["gradnorm_plateau_eps%.2f" % eps] = None if f is None else m[a + f]["epoch"]
                # gradient alignment with the next resolution
                nxt = [r_["resolution_next"] for r_ in m[seg]]
                cosv = np.array([role(r_, "next").get("cos_with_current", np.nan) for r_ in m[seg]])
                for c in (0.9, 0.8, 0.7, 0.5):
                    idx = np.flatnonzero(cosv < c)
                    fires["cos_next_below_%.1f" % c] = None if idx.size == 0 else m[a + int(idx[0])]["epoch"]
                # look-ahead: BN-recalibrated CE at the next resolution stops improving
                la = np.array([role(r_, "next").get("monitor_ce_bnrecal", np.nan) for r_ in m[seg]])
                for eps in (0.02, 0.05):
                    f = relative_decrease_trigger(la, eps, blackout=0)
                    fires["lookahead_plateau_eps%.2f" % eps] = None if f is None else m[a + f]["epoch"]
                # look-ahead rises (coarse training starts hurting the finer problem)
                rise = np.flatnonzero(np.diff(la) > 0)
                fires["lookahead_ce_rises"] = None if rise.size == 0 else m[a + int(rise[0]) + 1]["epoch"]
                rep["stage%d_r%d_switch_at_epoch%d" % (si, r, m[b - 1]["epoch"])] = fires
        report["go_no_go"][label] = gn
        report["spikes"][label] = sp
        report["rank_corr"][label] = rc
        report["trigger_replay"][label] = rep
        # Q-04
        q = {}
        for r_ in m:
            s32 = role(r_, "target")
            if r_["resolution_current"] != 32 and "test_acc_bnrecal" in s32:
                q[r_["epoch"]] = {"test_acc_current_path": r_["test_acc_current"],
                                  "test_acc_target_path": r_["test_acc_target"],
                                  "test_acc_target_bnrecal": s32["test_acc_bnrecal"]}
        report["q04_bn_share"][label] = q

    # 5b. step-size controller replay (phase 1 traces): largest Delta with tau >= theta
    lines.append("\n## 5b. Step-size controller replay: largest candidate step with tau >= theta (train-mode BN)\n")
    lines.append("| run | epoch | r | tau by candidate | chosen Delta (theta=0.5) | (theta=0.3) |\n|---|---:|---:|---|---:|---:|")
    for label, run in sorted(runs.items()):
        for r_ in run["metrics"]:
            tau = r_["signals"].get("tau_train")
            if not tau:
                continue
            cands = sorted(((v["delta"], v["tau"]) for v in tau.values() if v["tau"] is not None))
            pick = {th: max([d for d, t in cands if t >= th], default=0) for th in (0.5, 0.3)}
            report.setdefault("step_size_replay", {}).setdefault(label, []).append(
                {"epoch": r_["epoch"], "r": r_["resolution_current"], "tau": cands, "pick": pick})
            lines.append("| %s | %d | %d | %s | %d | %d |" % (
                label, r_["epoch"], r_["resolution_current"],
                " ".join("+%d:%.2f" % (d, t) for d, t in cands), pick[0.5], pick[0.3]))

    # 6. grid
    lines.append("\n## 6. Final test accuracy by configuration (all seeds present)\n")
    lines.append("| configuration | seeds | acc per seed | mean | test CE mean | train s |\n|---|---|---|---:|---:|---:|")
    by_cfg = {}
    for label, run in runs.items():
        s = run["summary"]
        if s:
            by_cfg.setdefault(label.rsplit("__seed", 1)[0], []).append(
                (s["seed"], s["final_test_acc"], s["final_test_ce"], s["train_seconds"]))
    rows = []
    for cfg, cells in by_cfg.items():
        cells.sort()
        mean = sum(c[1] for c in cells) / len(cells)
        report["grid"][cfg] = {"cells": cells, "mean_acc": mean}
        rows.append((mean, cfg, cells))
    for mean, cfg, cells in sorted(rows, reverse=True):
        lines.append("| %s | %s | %s | %.4f | %.4f | %.0f |" % (
            cfg, "/".join(str(c[0]) for c in cells), " / ".join("%.4f" % c[1] for c in cells),
            mean, sum(c[2] for c in cells) / len(cells), sum(c[3] for c in cells) / len(cells)))

    # 2. go/no-go table
    lines.append("\n## 2. Go / no-go rule of plan section 11.4 (D = relative decrease of EMA ||g|| over the last two epochs of a stage)\n")
    lines.append("| run | stage | D | verdict |\n|---|---|---:|---|")
    for label, gn in report["go_no_go"].items():
        for k, v in gn.items():
            lines.append("| %s | %s | %.3f | %s |" % (label, k, v["D_last2"], v["verdict"]))

    lines.append("\n## 3. Post-switch spikes of ||g|| (ratio to the last pre-switch check; length = checks above it)\n")
    lines.append("| run | event | amplitude | length |\n|---|---|---:|---:|")
    for label, sp in report["spikes"].items():
        for k, v in sp.items():
            lines.append("| %s | %s | %.2f | %d |" % (label, k, v["amplitude_ratio"], v["length_checks"]))

    lines.append("\n## 4. Spearman rank correlation of ||g|| with monitor CE inside a stage\n")
    lines.append("| run | stage | rho | n |\n|---|---|---:|---:|")
    for label, rc in report["rank_corr"].items():
        for k, v in rc.items():
            lines.append("| %s | %s | %.3f | %d |" % (label, k, v["spearman_g_vs_ce"], v["n"]))

    lines.append("\n## 5. Trigger replay: first epoch each criterion would fire (fixed switch at the stage's last epoch)\n")
    for label, rep in report["trigger_replay"].items():
        for k, fires in rep.items():
            lines.append("- **%s / %s**: %s" % (label, k, ", ".join("%s=%s" % (a, b) for a, b in fires.items())))

    lines.append("\n## 7. Q-04: target-path (32x32) test accuracy while training at a lower resolution\n")
    lines.append("| run | epoch | current path | target path | target after BN recal |\n|---|---:|---:|---:|---:|")
    for label, q in report["q04_bn_share"].items():
        for e, v in q.items():
            lines.append("| %s | %d | %.4f | %.4f | %.4f |" % (label, e, v["test_acc_current_path"],
                                                          v["test_acc_target_path"], v["test_acc_target_bnrecal"]))

    Path(out_json).write_text(json.dumps(report, indent=1))
    md = Path(out_json).with_suffix(".md")
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote", out_json, "and", md)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="adaptive-phase*")
    ap.add_argument("--out", default=str(ROOT / "results" / "adaptive_phase0_analysis.json"))
    args = ap.parse_args()
    runs = load(args.slug)
    print("loaded %d runs" % len(runs))
    analyse(runs, args.out)

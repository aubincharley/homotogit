"""Shared loading, statistics and figure helpers for the v3 analysis."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import common as C
from . import v2points

FIG = C.STUDY / "figures"
TAB = C.STUDY / "tables"
RAW = C.STUDY / "raw"
COL = {"plain": "#444444", "resolution_max_b1": "#1f77b4", "gaussian_postrelu": "#ff7f0e",
       "resolution_max_b1_gaussian_conv": "#9467bd"}
LAB = {"plain": "plain", "resolution_max_b1": "resolution-only", "gaussian_postrelu": "Gaussian-only",
       "resolution_max_b1_gaussian_conv": "combined"}
SPL = {"train_probe": "train probe (1,000)", "test_probe": "test probe (1,000)",
       "train_large": "train subset (10,000)", "test_full": "full test set (10,000)"}
POL = C.POLICY_LABEL
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "figure.dpi": 150})
ELEV, AZIM, BOX = 28, -58, (1, 1, 0.75)


class Data:
    """All points: v3 raw (every account, pilots excluded) + mapped v2 points."""

    def __init__(self, raw=RAW):
        self.raw = raw
        self.dirs = sorted(p for p in raw.glob("*/v3") if p.is_dir() and not p.parent.name.endswith("_pilot"))
        self.rows, self.meta, self.issues = {}, {}, []
        self.extra = defaultdict(dict)          # task-kind specific non-point rows (hchecks, quadform)
        for d in self.dirs:
            for p in sorted((d / "eval").glob("*.jsonl")):
                tid = p.name[:-6]
                for line in p.read_text().splitlines():
                    try:
                        r = json.loads(line)
                    except Exception:
                        self.issues.append({"task": tid, "problem": "unparsable line"})
                        continue
                    k = r["key"]
                    if k.startswith("hchecks|") or k.startswith("quad|"):
                        self.extra[tid][k] = r
                    else:
                        r["_source"] = "v3:" + tid
                        self.rows[k] = r
            for p in sorted((d / "eval").glob("*.meta.json")):
                self.meta[p.name[:-10]] = json.loads(p.read_text())
        self.v2 = v2points.load()
        self.n_v3, self.n_v2_used = len(self.rows), 0
        for k, v in self.v2.items():
            if k not in self.rows:
                self.rows[k] = v
                self.n_v2_used += 1

    def ce(self, key, split):
        r = self.rows.get(key)
        return None if r is None else r["splits"][split]["ce"]

    def acc(self, key, split):
        r = self.rows.get(key)
        return None if r is None else r["splits"][split]["acc"]

    def hessian(self):
        out = {}
        for d in self.dirs:
            for p in sorted((d / "hessian").glob("*.json")):
                out[p.stem] = json.loads(p.read_text())
        return out

    def vectors(self, pid):
        import torch
        for d in self.dirs:
            p = d / "hessian" / (pid + "_vectors.pt")
            if p.exists():
                return torch.load(p, map_location="cpu", weights_only=True)
        return None

    def status_counts(self):
        c = defaultdict(int)
        bad = []
        for tid, m in self.meta.items():
            c[m["status"]] += 1
            if m["status"] != "complete":
                bad.append({"task": tid, "status": m["status"], "traceback": (m.get("traceback") or m.get("reason") or "")[-800:]})
        return dict(c), bad


def S_of(D, m, s, ckpt, state, pol, splitset, split, k, amp, prec="f32"):
    c = D.ce(C.pkey(m, s, ckpt, state, pol, splitset, "c", prec), split)
    p = D.ce(C.pkey(m, s, ckpt, state, pol, splitset, C.spec_r1(k, amp), prec), split)
    q = D.ce(C.pkey(m, s, ckpt, state, pol, splitset, C.spec_r1(k, -amp), prec), split)
    if c is None or p is None or q is None:
        return None
    return {"L0": c, "Lp": p, "Lm": q, "S": 0.5 * (p + q) - c}


def seedstats(vals: dict):
    v = [vals[s] for s in C.SEEDS if vals.get(s) is not None and np.isfinite(vals[s])]
    out = {"n_seeds": len(v), "mean": float(np.mean(v)) if v else float("nan"),
           "sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
           "n_negative": int(sum(x < 0 for x in v)), "n_positive": int(sum(x > 0 for x in v)),
           "n_zero": int(sum(x == 0 for x in v))}
    for s in C.SEEDS:
        out["seed%d" % s] = vals.get(s, float("nan"))
    return out


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    except ValueError:
        # a log axis without positive data (e.g. an empty panel): draw it linear and say so
        for ax in fig.axes:
            if getattr(ax, "get_yscale", None) and ax.get_yscale() == "log":
                ax.set_yscale("linear")
                ax.text(0.02, 0.02, "linear axis: no positive data", transform=ax.transAxes, fontsize=6, color="0.4")
        fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(FIG / (name + ".png"), bbox_inches="tight", dpi=170)
    plt.close(fig)


def write_csv(name, rows):
    TAB.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(TAB / (name + ".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items()})


def state_label(st):
    if st is None:
        return "-"
    parts = []
    if st.get("resolution") is not None:
        parts.append("r=%d" % st["resolution"])
    if st.get("sigma") is not None:
        parts.append("G=%s" % C.fnum(st["sigma"]))
    return ", ".join(parts) if parts else "plain"


def mark_transitions(ax, ymin_frac=0.0):
    for e in (6, 12):
        ax.axvline(e, color="#1f77b4", ls=":", lw=0.8, alpha=0.8)
    for e in (3, 6, 9, 12, 15, 18, 21):
        ax.axvline(e, color="#999999", ls="-", lw=0.4, alpha=0.5)

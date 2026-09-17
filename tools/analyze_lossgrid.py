"""Train/test loss and accuracy for every method under every objective.

Reads the objective-grid shards and prints one table per objective, plus the
contrast against the plain control that the whole study turns on.

Two reading rules are enforced here rather than left to the reader.

**Accuracy is the comparator.**  Each arm optimises a different quantity, so
``obj`` is not comparable across columns -- a square-loss value and a focal value
are not on one scale.  Cross-entropy is reported for every run whatever was
trained, so ``ce`` is the common scale; accuracy needs no scale at all.

**The square loss is read at its best learning rate.**  Its gradient does not
share cross-entropy's scale, so a single learning rate would confound the
objective with the step size.  ``--lr-policy best`` (the default) takes, per
objective, the learning rate with the lowest mean test error across the four
methods -- chosen once for the arm, never per method, so the comparison between
methods inside an arm stays paired.

    python tools/analyze_lossgrid.py results/kaggle_outputs
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

METHODS = ("plain", "resolution_max_b1", "gaussian_postrelu",
           "resolution_max_b1_gaussian_conv")
SHORT = {"plain": "plain", "resolution_max_b1": "R",
         "gaussian_postrelu": "G", "resolution_max_b1_gaussian_conv": "RG"}
ARM_LABEL = {"ce": "cross-entropy", "ls": "label smoothing 0.1",
             "focal": "focal (gamma=2)", "square": "square loss on logits"}


def load(roots) -> list:
    rows = []
    for root in roots:
        for path in sorted(glob.glob(str(Path(root) / "**" / "loss_shard*.json"),
                                     recursive=True)):
            for c in json.loads(Path(path).read_text())["cells"]:
                method, arm, lr, seed = c["cell"].split("__")
                final = c["final"]["target"]
                rows.append({"method": method, "arm": arm,
                             "lr": float(lr[2:]), "seed": int(seed[4:]),
                             "train_acc": final["train_probe"]["acc"],
                             "test_acc": final["test"]["acc"],
                             "train_ce": final["train_probe"]["ce"],
                             "test_ce": final["test"]["ce"],
                             "train_obj": final["train_probe"].get("obj"),
                             "test_obj": final["test"].get("obj")})
    return rows


def load_reference_ce(roots) -> list:
    """The recorded cross-entropy cells, reusable as the control.

    This branch's change is a no-op for cross-entropy, so a ``ce`` arm need not
    be re-run; the reference cells of the 28-cell grid are the same runs.  They
    carry no ``obj`` field, which is correct -- for cross-entropy it equals ``ce``.
    """
    rows = []
    for root in roots:
        for path in sorted(glob.glob(str(Path(root) / "**" / "grid_shard*.json"),
                                     recursive=True)):
            for c in json.loads(Path(path).read_text())["cells"]:
                if "__reference__" not in c["cell"]:
                    continue
                method, _, seed = c["cell"].split("__")
                final = c["final"]["target"]
                rows.append({"method": method, "arm": "ce", "lr": 0.005,
                             "seed": int(seed[4:]),
                             "train_acc": final["train_probe"]["acc"],
                             "test_acc": final["test"]["acc"],
                             "train_ce": final["train_probe"]["ce"],
                             "test_ce": final["test"]["ce"],
                             "train_obj": final["train_probe"]["ce"],
                             "test_obj": final["test"]["ce"]})
    return rows


def agg(rows, keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return st.fmean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def pick_lr(rows, arm) -> float:
    """The learning rate with the lowest mean test error over the four methods."""
    per_lr = agg([r for r in rows if r["arm"] == arm], ["lr"])
    scored = {lr[0]: st.fmean(r["test_acc"] for r in rs) for lr, rs in per_lr.items()}
    return max(scored, key=scored.get)


def table(rows, arm, lr) -> str:
    sel = [r for r in rows if r["arm"] == arm and r["lr"] == lr]
    by_method = agg(sel, ["method"])
    lines = ["", "=== %s   (lr = %g, n = %d seeds) ===" %
             (ARM_LABEL.get(arm, arm), lr,
              max((len(v) for v in by_method.values()), default=0)),
             "%-8s %16s %10s %10s %10s %10s" %
             ("method", "test acc", "train acc", "test CE", "train CE", "test obj")]
    ctrl = None
    for m in METHODS:
        rs = by_method.get((m,))
        if not rs:
            continue
        ta, tasd = mean_sd([r["test_acc"] for r in rs])
        ra, _ = mean_sd([r["train_acc"] for r in rs])
        tc, _ = mean_sd([r["test_ce"] for r in rs])
        rc, _ = mean_sd([r["train_ce"] for r in rs])
        to, _ = mean_sd([r["test_obj"] for r in rs])
        if m == "plain":
            ctrl = ta
        delta = "" if ctrl is None or m == "plain" else "  (%+.2f pp)" % (100 * (ta - ctrl))
        lines.append("%-8s %8.2f +- %-5.2f %9.2f %10.4f %10.4f %10s%s" %
                     (SHORT[m], 100 * ta, 100 * tasd, 100 * ra, tc, rc,
                      "%.4f" % to if to is not None else "-", delta))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--lr-policy", choices=("best", "reference"), default="best")
    ap.add_argument("--json", help="write the aggregated rows here")
    a = ap.parse_args()

    rows = load(a.roots) + load_reference_ce(a.roots)
    if not rows:
        raise SystemExit("no loss_shard*.json or grid_shard*.json under %s" % (a.roots,))

    arms = [x for x in ("ce", "ls", "focal", "square")
            if any(r["arm"] == x for r in rows)]
    chosen = {}
    for arm in arms:
        lrs = sorted({r["lr"] for r in rows if r["arm"] == arm})
        chosen[arm] = (pick_lr(rows, arm) if a.lr_policy == "best" and len(lrs) > 1
                       else (0.005 if 0.005 in lrs else lrs[0]))
        print(table(rows, arm, chosen[arm]))
        if len(lrs) > 1:
            print("    learning rates run: %s; shown: %g (%s)" %
                  (", ".join("%g" % v for v in lrs), chosen[arm], a.lr_policy))

    print("\n=== the contrast, per objective (test accuracy vs plain) ===")
    print("%-22s %10s %10s %10s" % ("objective", "R", "G", "RG"))
    for arm in arms:
        sel = agg([r for r in rows if r["arm"] == arm and r["lr"] == chosen[arm]],
                  ["method"])
        base = mean_sd([r["test_acc"] for r in sel.get(("plain",), [])])[0]
        if base is None:
            continue
        cells = []
        for m in METHODS[1:]:
            v = mean_sd([r["test_acc"] for r in sel.get((m,), [])])[0]
            cells.append("%+9.2f" % (100 * (v - base)) if v is not None else "        -")
        print("%-22s %s" % (ARM_LABEL.get(arm, arm), " ".join(cells)))
    print("\naccuracy is the comparator; obj is not comparable across objectives,")
    print("CE is reported for every run whatever was trained.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"rows": rows, "lr_chosen": chosen, "lr_policy": a.lr_policy}, indent=2))
        print("wrote %s" % a.json)


if __name__ == "__main__":
    main()

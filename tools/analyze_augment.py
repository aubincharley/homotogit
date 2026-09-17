"""What the curriculum is worth, corner by corner, ordered by baseline strength.

    python tools/analyze_augment.py results/kaggle_outputs
    python tools/analyze_augment.py results/kaggle_outputs /tmp/a60 --synthesis

Four reading rules are enforced here rather than left to the reader.

**Accuracy is the comparator.** Every corner trains the same four methods on the
same data with the same seeds; the only thing that differs inside a corner is the
operator, so the difference in test accuracy is the quantity of interest. Cross-
entropy is reported alongside but is not comparable across objectives.

**A corner includes whether the schedule was stretched.** Schedules are indexed
by absolute epoch, so a 60-epoch run either keeps its boundaries -- the operator
reaches its target state at the same epoch as at 30 and the rest trains the bare
network -- or has them multiplied so the operator keeps its share of training.
Those are different experiments. Cells that disagree are never pooled, and a
corner that somehow contains both is reported as an error rather than averaged.

**Rows are ordered by baseline, because that is the finding.** The gain is worth
four to five points when the control sits below 79 %, about one point between 80
and 85 %, and nothing above that -- and the same erosion appears whether the
baseline was raised by changing the objective or by augmenting the data.

**The budget control is named.** Comparing `crop_flip @ 60` against
`crop_flip @ 30` alone would confound augmentation with budget *and* with the
curriculum's shrinking share of training. `none @ 60` at the same learning rate
is what separates them, so the report calls it out explicitly when present.
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
SHORT = {"plain": "plain", "resolution_max_b1": "R", "gaussian_postrelu": "G",
         "resolution_max_b1_gaussian_conv": "RG"}


def _rows_from_augment(roots) -> list:
    rows = []
    for root in roots:
        for path in sorted(glob.glob(str(Path(root) / "**" / "augment_shard*.json"),
                                     recursive=True)):
            blob = json.loads(Path(path).read_text())
            for c in blob["cells"]:
                method, aug, epochs, lr, seed = c["cell"].split("__")
                f = c["final"]["target"]
                rows.append({"corner": (aug, int(epochs[1:]), float(lr[2:]),
                                        bool(c.get("stretched", blob.get("stretched", False)))),
                             "method": method, "seed": int(seed[4:]),
                             "test": f["test"]["acc"], "train": f["train_probe"]["acc"],
                             "test_ce": f["test"]["ce"]})
    return rows


def _rows_from_reference(roots) -> list:
    """The recorded 30-epoch cells: `none`, lr 0.005, and the recipe variants."""
    lr_of = {"reference": 0.005, "lr_low": 0.0025, "lr_high": 0.01}
    rows = []
    for root in roots:
        for path in sorted(glob.glob(str(Path(root) / "**" / "grid_shard*.json"),
                                     recursive=True)):
            for c in json.loads(Path(path).read_text())["cells"]:
                method, variant, seed = c["cell"].split("__")
                if variant not in lr_of:
                    continue
                f = c["final"]["target"]
                rows.append({"corner": ("none", 30, lr_of[variant], False),
                             "method": method, "seed": int(seed[4:]),
                             "test": f["test"]["acc"], "train": f["train_probe"]["acc"],
                             "test_ce": f["test"]["ce"]})
    return rows


def _rows_from_objectives(roots) -> list:
    """The objective grid, folded in as extra ways of moving the baseline."""
    rows = []
    for root in roots:
        for path in sorted(glob.glob(str(Path(root) / "**" / "loss_shard*.json"),
                                     recursive=True)):
            for c in json.loads(Path(path).read_text())["cells"]:
                method, arm, lr, seed = c["cell"].split("__")
                f = c["final"]["target"]
                rows.append({"corner": ("loss:" + arm, 30, float(lr[2:]), False),
                             "method": method, "seed": int(seed[4:]),
                             "test": f["test"]["acc"], "train": f["train_probe"]["acc"],
                             "test_ce": f["test"]["ce"]})
    return rows


def dedupe(rows) -> tuple:
    """One row per ``(corner, method, seed)``, keeping the first seen.

    The same cell legitimately exists in more than one place -- a partially
    retrieved kernel output and a complete copy of it, say -- and pooling both
    would weight that seed twice without any sign of it in the output.  A cell
    read twice is dropped silently only if the two copies agree; if they differ
    the run is ambiguous and the caller is told rather than averaged over.
    """
    seen, out, dups, conflicts = {}, [], 0, []
    for r in rows:
        key = (r["corner"], r["method"], r["seed"])
        if key in seen:
            dups += 1
            if abs(seen[key]["test"] - r["test"]) > 1e-9:
                conflicts.append(key)
            continue
        seen[key] = r
        out.append(r)
    return out, dups, conflicts


def label(corner) -> str:
    aug, epochs, lr, stretched = corner
    name = aug[5:] if aug.startswith("loss:") else ("aug " + aug if aug != "none" else "aucune")
    out = "%s, %de, lr %g" % (name, epochs, lr)
    return out + (" [étiré]" if stretched else "")


def summarise(rows) -> list:
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["corner"]][r["method"]].append(r)
    out = []
    for corner, per_method in by.items():
        ctrl = per_method.get("plain")
        if not ctrl:
            continue
        base = st.fmean(r["test"] for r in ctrl)
        entry = {"corner": corner, "label": label(corner),
                 "base_test": 100 * base,
                 "base_train": 100 * st.fmean(r["train"] for r in ctrl),
                 "n_control": len(ctrl), "gains": {}, "n": {}}
        for m in METHODS[1:]:
            rs = per_method.get(m)
            if rs:
                entry["gains"][SHORT[m]] = 100 * (st.fmean(r["test"] for r in rs) - base)
                entry["n"][SHORT[m]] = len(rs)
        out.append(entry)
    out.sort(key=lambda e: e["base_test"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--synthesis", action="store_true",
                    help="fold in the objective grid as further ways of moving the baseline")
    ap.add_argument("--json", help="write the summary here")
    a = ap.parse_args()

    rows = _rows_from_augment(a.roots) + _rows_from_reference(a.roots)
    if a.synthesis:
        rows += _rows_from_objectives(a.roots)
    if not rows:
        raise SystemExit("no augment_shard*.json or grid_shard*.json under %s" % (a.roots,))

    rows, dups, conflicts = dedupe(rows)
    if conflicts:
        raise SystemExit("the same cell appears twice with different results: %s"
                         % (sorted(set(conflicts))[:5],))
    if dups:
        print("(%d cellules lues en double, ignorées)\n" % dups)

    table = summarise(rows)
    print("%-30s %10s %11s | %8s %8s %8s   %s"
          % ("réglage", "base test", "base train", "R", "G", "RG", "graines"))
    print("-" * 96)
    for e in table:
        cells = []
        for m in ("R", "G", "RG"):
            cells.append("%+8.2f" % e["gains"][m] if m in e["gains"] else "       -")
        ns = "/".join(str(e["n"].get(m, 0)) for m in ("R", "G", "RG"))
        print("%-30s %9.2f%% %10.2f%% | %s %s %s   %s (ctrl %d)"
              % (e["label"], e["base_test"], e["base_train"], *cells, ns, e["n_control"]))

    # the budget control, named rather than left to be inferred
    corners = {e["corner"]: e for e in table}
    for lr in sorted({c[2] for c in corners}):
        aug60, non60 = ("crop_flip", 60, lr, False), ("none", 60, lr, False)
        if aug60 in corners and non60 in corners:
            print("\ncontrôle de budget à lr %g, 60 époques, curriculum dilué de la même façon :" % lr)
            for m in ("R", "G", "RG"):
                g_aug = corners[aug60]["gains"].get(m)
                g_non = corners[non60]["gains"].get(m)
                if g_aug is None or g_non is None:
                    continue
                print("   %-3s sans augmentation %+6.2f  ->  avec %+6.2f   (%+.2f)"
                      % (m, g_non, g_aug, g_aug - g_non))
            print("   un gain qui survit sans augmentation et disparaît avec elle "
                  "n'est pas un effet de dilution.")

    print("\nl'exactitude test est le comparateur ; la CE n'est pas comparable entre objectifs.")
    if a.json:
        Path(a.json).write_text(json.dumps(
            [{k: (list(v) if k == "corner" else v) for k, v in e.items()} for e in table],
            indent=2, ensure_ascii=False))
        print("écrit %s" % a.json)


if __name__ == "__main__":
    main()

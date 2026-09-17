"""Rebuild `docs/curves/data.json` from the recorded runs.

Everything the curves page draws comes from here, so a panel can never disagree
with what `analyze_augment.py` and `analyze_lossgrid.py` report from the same
files.  Run it after new shards land:

    python tools/build_curves_data.py

Each arm is one training setting; a cell is averaged over its seeds.  Cells are
deduplicated on `(arm, method, seed)` because the same run legitimately exists in
more than one retrieved kernel output, and pooling both would weight that seed
twice with nothing in the output to show for it.

Evaluation is read on the **target** path -- the operator at identity -- so the
four arms are compared on the same class of network at every epoch, not only at
the end.
"""
from __future__ import annotations

import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "kaggle_outputs"

LABELS = {
    "ce": "CE · 30e · lr 0,005",
    "ls": "label smoothing · 30e",
    "focal": "focal γ=2 · 30e",
    "square": "square loss · 30e · lr 0,1",
    "crop_flipe30@0.005": "augmentation · 30e · lr 0,005",
    "crop_flipe30@0.01": "augmentation · 30e · lr 0,01",
    "nonee60@0.01": "SANS augmentation · 60e · lr 0,01",
    "crop_flipe60@0.005": "augmentation · 60e · lr 0,005",
    "crop_flipe60@0.01": "augmentation · 60e · lr 0,01",
}
ORDER = list(LABELS)
GAINERS = (("resolution_max_b1", "R"), ("gaussian_postrelu", "G"),
           ("resolution_max_b1_gaussian_conv", "RG"))


def collect() -> dict:
    per = defaultdict(lambda: defaultdict(list))
    seen = set()

    def add(arm, method, seed, path):
        if (arm, method, seed) in seen:
            return
        seen.add((arm, method, seed))
        for rec in json.loads(Path(path).read_text()):
            t = rec["eval"]["target"]
            per[(arm, method)][rec["epoch"]].append(
                (t["test"]["acc"], t["train_probe"]["acc"]))

    for p in glob.glob(str(OUT / "cc-loss-*/runs/*/metrics.json")):
        m, arm, lr, s = Path(p).parent.name.split("__")
        if arm != "square":                      # the square cells of that grid
            add(arm, m, s, p)                    # never converged; see OBJECTIVES.md
    for p in glob.glob(str(OUT / "cc-sq-*/runs/*/metrics.json")):
        m, arm, lr, s = Path(p).parent.name.split("__")
        if float(lr[2:]) == 0.1:                 # the square loss at its optimum
            add("square", m, s, p)
    for p in glob.glob(str(OUT / "cc-grid-*/runs/*/metrics.json")):
        m, variant, s = Path(p).parent.name.split("__")
        if variant == "reference":
            add("ce", m, s, p)
    for pattern in ("cc-aug-*/runs/*/metrics.json", "cc-a60-*/runs/*/metrics.json"):
        for p in glob.glob(str(OUT / pattern)):
            m, aug, ep, lr, s = Path(p).parent.name.split("__")
            add("%s%s@%g" % (aug, ep, float(lr[2:])), m, s, p)
    return per


def main() -> None:
    per = collect()
    if not per:
        raise SystemExit("no metrics under %s" % OUT)

    data = {}
    for (arm, method), by_epoch in per.items():
        eps = sorted(by_epoch)
        data.setdefault(arm, {})[method] = {
            "epochs": eps, "n": len(by_epoch[eps[0]]),
            "test_acc": [round(st.fmean(x[0] for x in by_epoch[e]), 5) for e in eps],
            "train_acc": [round(st.fmean(x[1] for x in by_epoch[e]), 5) for e in eps]}

    scatter = []
    for arm in data:
        if "plain" not in data[arm]:
            continue
        base = data[arm]["plain"]["test_acc"][-1]
        for mid, short in GAINERS:
            if mid in data[arm]:
                scatter.append({"arm": LABELS.get(arm, arm), "m": short,
                                "base": round(100 * base, 2),
                                "gain": round(100 * (data[arm][mid]["test_acc"][-1] - base), 2)})
    data["_scatter"] = scatter
    data["_labels"] = LABELS

    dst = ROOT / "docs" / "curves" / "data.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    print("%s  %d arms, %d scatter points, %d bytes"
          % (dst.relative_to(ROOT), len(data) - 2, len(scatter), dst.stat().st_size))
    for arm in ORDER:
        if arm in data:
            n = {m: data[arm][m]["n"] for m in data[arm]}
            print("  %-22s %d methods, seeds %s, %d epochs"
                  % (arm, len(data[arm]), sorted(set(n.values())),
                     len(data[arm]["plain"]["epochs"]) - 1))


if __name__ == "__main__":
    main()

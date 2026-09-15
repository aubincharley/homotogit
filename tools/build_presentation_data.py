"""Rebuild `docs/presentation/data.json` from the probe and grid results.

Everything the presentation draws comes from here, so a figure can never
disagree with `results/grid_report.json`.  Only the 12 reference cells
(4 methods x 3 seeds) feed the per-method curves; the 20 conditions feed the
dial test and its scatter.

Four blocks are transcribed rather than computed, because they come from runs
whose raw output lives elsewhere: the three-source accuracy table and the
BatchNorm 2x2 (`docs/CROSS_STUDY.md`), the PAC-Bayes distances
(`docs/RESULTS_GRID.md` section 5), and the schedules (`docs/METHODS.md`).

    python tools/build_presentation_data.py
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("plain", "gaussian_postrelu", "resolution_max_b1",
           "resolution_max_b1_gaussian_conv")
LABEL = {"plain": "plain", "gaussian_postrelu": "flou",
         "resolution_max_b1": "résolution",
         "resolution_max_b1_gaussian_conv": "combiné"}
BLOCKS = ["stem"] + ["block%d" % i for i in range(9)] + ["fc"]


def load_cells() -> dict:
    cells: dict = {}
    for path in sorted(glob.glob(str(ROOT / "results/kaggle_outputs/cc-probe-s*/probe_shard*.json"))):
        cells.update(json.loads(Path(path).read_text())["cells"])
    if not cells:
        raise SystemExit("no probe shards under results/kaggle_outputs/")
    return cells


def per_method(cells: dict) -> list:
    out = []
    for name in METHODS:
        rec = [cells[k] for k in sorted(cells) if k.startswith(name + "__reference__")]
        if len(rec) != 3:
            raise SystemExit("%s: expected 3 reference seeds, found %d" % (name, len(rec)))

        def mean(get):
            return st.fmean(get(r) for r in rec)

        def curve(get):
            arrays = [get(r) for r in rec]
            n = min(len(a) for a in arrays)
            return [round(st.fmean(a[i] for a in arrays), 8) for i in range(n)]

        sens = lambda r: r["sensitivity"]           # noqa: E731
        out.append({
            "id": name, "label": LABEL[name], "full": name,
            "test_err": round(mean(lambda r: r["test"]["err"]), 4),
            "train_err": round(mean(lambda r: r["train"]["err"]), 4),
            "test_ce": round(mean(lambda r: r["test"]["ce"]), 4),
            "train_ce": round(mean(lambda r: r["train"]["ce"]), 4),
            "trH": round(mean(lambda r: r["curvature"]["trace"]["mean"]), 1),
            "trH_sem": round(mean(lambda r: r["curvature"]["trace"]["sem"]), 1),
            "top_share": round(mean(lambda r: r["curvature"]["top_share_of_trace"]), 3),
            "radius": round(mean(lambda r: sens(r)["frequency"]["mean_radius"]), 3),
            "jf": round(mean(lambda r: sens(r)["jacobian"]["frobenius_mean"]), 1),
            "js": round(mean(lambda r: sens(r)["jacobian"]["spectral_mean"]), 1),
            "ring": curve(lambda r: sens(r)["frequency"]["ring_energy_per_mode"]),
            "sv": curve(lambda r: sens(r)["jacobian"]["singular_values_mean"]),
            "S": curve(lambda r: [d["displacement"] for d in sens(r)["amplitude_curve"]]),
            "acc": curve(lambda r: [d["accuracy"] for d in sens(r)["amplitude_curve"]]),
            "lin": curve(lambda r: [d["ratio"] for d in sens(r)["linearity_ratio"]]),
            "blocks": [round(mean(lambda r: r["curvature"]["per_block"][b]["mean"]), 1)
                       for b in BLOCKS],
        })
    return out


def main() -> None:
    cells = load_cells()
    report = json.loads((ROOT / "results/grid_report.json").read_text())
    methods = per_method(cells)
    eps = [d["epsilon"] for d in
           cells[METHODS[0] + "__reference__seed0"]["sensitivity"]["amplitude_curve"]]

    rows = sorted(report["dial_test"]["rows"], key=lambda r: -abs(r["partial"]))
    keep = ("condition", "method", "variant", "group", "train_error", "test_error",
            "gap", "trace_H", "jac_frobenius", "mean_radius", "S@3", "S@0.05")

    data = {
        "methods": methods,
        "eps": eps,
        "blockNames": BLOCKS,
        # -- transcribed, see the module docstring --
        "acc3": [
            {"m": "plain", "probes": [75.93, 1.04], "land": [75.65, 0.78], "methods": [75.43, 0.76]},
            {"m": "résolution", "probes": [80.22, 0.01], "land": [80.21, 0.36], "methods": [80.37, 0.61]},
            {"m": "flou", "probes": [79.95, 0.04], "land": [80.12, 0.49], "methods": [79.91, 0.27]},
            {"m": "combiné", "probes": [81.35, 0.09], "land": [81.58, 0.30], "methods": [81.51, 0.22]},
        ],
        "bnCols": ["stats gelées / train", "stats gelées / test",
                   "recalibré / train", "recalibré / test"],
        "bn": [{"m": "flou", "v": [-30.3, -42.6, -19.2, -21.1]},
               {"m": "résolution", "v": [-32.0, -46.0, -28.1, -30.5]},
               {"m": "combiné", "v": [-41.0, -54.9, -36.9, -41.6]}],
        "pac": [{"m": "plain", "e1": 45.1, "e6": 31.92, "e12": 16.63},
                {"m": "résolution", "e1": 47.7, "e6": 35.92, "e12": 26.07},
                {"m": "flou", "e1": 28.2, "e6": 19.37, "e12": 14.23},
                {"m": "combiné", "e1": 30.2, "e6": 22.71, "e12": 13.61}],
        "pacTotal": 13.6,
        # -- computed --
        "dial": [{"f": r["field"], "pooled": round(r["partial"], 3),
                  "ctrl": round(r["groups"]["within"]["control"]["partial"], 3),
                  "curr": round(r["groups"]["within"]["curriculum"]["partial"], 3),
                  "wpool": round(r["groups"]["within_pooled"], 3),
                  "art": r["groups"]["grouping_artefact"]} for r in rows[:8]],
        "conditions": [{"c": c["condition"], "g": c["group"], "m": c["method"],
                        "v": c["variant"], "te": round(c["test_error"], 4),
                        "tr": round(c["train_error"], 4),
                        "s005": round(c["S@0.05"], 4), "jf": round(c["jac_frobenius"], 2),
                        "trH": round(c["trace_H"], 1), "rad": round(c["mean_radius"], 3)}
                       for c in report["conditions"] if set(keep) <= set(c)],
        "ceiling": round(report["detection_ceiling"]["ceiling"], 3),
    }

    out = ROOT / "docs/presentation/data.json"
    out.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    print("%s  %d conditions, %d methods, %d bytes"
          % (out.relative_to(ROOT), len(data["conditions"]), len(methods), out.stat().st_size))


if __name__ == "__main__":
    main()

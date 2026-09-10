"""Aggregate the anti-aliasing ablation into one table, one comparison set, one JSON.

Reads every ``summary.json`` and ``diagnostics.json`` produced by the four
ablation jobs and writes ``results/ablation_aa_results.json``.

Interpretation rules, applied here rather than left to the reader:

* an arm's headline number is its ``final_test_acc_primary`` -- the **current**
  path for the constant-sigma arms (which never reach the target endpoint) and
  the **target** path for every arm that does.  Reading the target path of a
  constant-sigma arm reports a premature ablation with BatchNorm mismatch, not
  predictor quality;
* one seed: differences are signs and magnitudes, never significance.  No SD
  exists.  Errata C-12/C-13 apply -- a single run defines no distribution;
* nothing here is comparable to ``results/campaign_results.json``: the initial
  weights differ (see ``scripts/stage_ablation_assets.py``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUTS = ROOT / "results" / "kaggle_outputs"

from scripts.ablation_manifest import build_configs as _w1  # noqa: E402
from scripts.ablation2_manifest import build_configs as _w2  # noqa: E402
from scripts.ablation3_manifest import build_configs as _w3  # noqa: E402
from scripts.ablation4_manifest import build_configs as _w4  # noqa: E402  (seeds only)


def build_configs():
    """All three waves: 12 at constant 32x32, 6 on the resolution row, 6 on the
    depth-of-reduction axis."""
    return _w1() + _w2() + _w3()


def _stats(values):
    """Mean and *descriptive* sample SD across seeds -- not a confidence statement."""
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return None, None
    mean = sum(vals) / n
    if n < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, var ** 0.5


def collect() -> dict:
    """One entry per configuration, aggregating however many seeds it has.

    Seeds are grouped rather than overwritten: waves 1-3 ran seed 0 only, wave 4
    adds seeds 1 and 2 to four arms, so the same table now holds 1-seed and
    3-seed entries side by side.  ``n_seeds`` says which is which, and a
    single-seed entry carries ``acc_sd = None`` rather than a fabricated zero.
    """
    cfgs = {c["id"]: c for c in build_configs()}
    runs = {}
    for summ in sorted(OUTPUTS.glob("abl*-j*/**/summary.json")):
        s = json.loads(summ.read_text())
        cid = s["cell_id"].split("__seed")[0]
        diag_f = summ.parent / "diagnostics.json"
        s["_diag"] = json.loads(diag_f.read_text()) if diag_f.is_file() else {}
        runs.setdefault(cid, []).append(s)

    arms = {}
    for cid, group in runs.items():
        group.sort(key=lambda s: s["seed"])
        acc = [s["final_test_acc_primary"] for s in group]
        acc_mean, acc_sd = _stats(acc)
        probe = [(s["final_train_probe_ce_current"]
                  if s["primary_path"] == "current" else s["final_train_probe_ce"])
                 for s in group]
        cons = [s["_diag"].get("shift_consistency", {}).get("consistency_mean")
                for s in group]
        alias = {}
        for key in ("blocks3", "blocks6"):
            vals = [s["_diag"].get("aliasing_energy", {}).get("measured_at", {})
                    .get(key, {}).get("alias_energy_fraction") for s in group]
            alias[key] = _stats(vals)[0]
        arms[cid] = {
            "n_seeds": len(group), "seeds": [s["seed"] for s in group],
            "primary_path": group[0]["primary_path"],
            "acc": acc_mean, "acc_sd": acc_sd, "acc_per_seed": acc,
            "acc_target": _stats([s["final_test_acc"] for s in group])[0],
            "acc_current": _stats([s["final_test_acc_current"] for s in group])[0],
            "ce_target": _stats([s["final_test_ce"] for s in group])[0],
            "probe_ce": _stats(probe)[0],
            "wall_s": _stats([s["wall_seconds"] for s in group])[0],
            "shift_consistency": _stats(cons)[0],
            "alias": alias,
            "rationale": cfgs.get(cid, {}).get("rationale", ""),
        }
    return arms


def compare(arms, a, b):
    """Paired comparison on the seeds the two arms actually share.

    Pairing is by seed, and only seeds present on both sides count: comparing a
    3-seed mean against a 1-seed value would silently mix a mean with a draw.
    ``all_same_sign`` is reported only when more than one seed is shared.
    """
    if a not in arms or b not in arms:
        return None
    A, B = arms[a], arms[b]
    shared = sorted(set(A["seeds"]) & set(B["seeds"]))
    if not shared:
        return None
    ia = {s: v for s, v in zip(A["seeds"], A["acc_per_seed"])}
    ib = {s: v for s, v in zip(B["seeds"], B["acc_per_seed"])}
    d = [(ia[s] - ib[s]) * 100 for s in shared]
    mean, sd = _stats(d)
    return {"a": a, "b": b, "seeds": shared, "n_seeds": len(shared),
            "d_acc_pp": round(mean, 2),
            "d_acc_sd": None if sd is None else round(sd, 2),
            "d_per_seed": [round(v, 2) for v in d],
            "all_same_sign": (all(v > 0 for v in d) or all(v < 0 for v in d))
                             if len(d) > 1 else None,
            "acc_a": A["acc"], "acc_b": B["acc"]}


COMPARISONS = [
    # wave 3 -- depth of the single reduction, and the blur at each depth
    ("D0", "R4"), ("D1", "R4"), ("D2", "R4"), ("D1", "R1"),
    ("D0G", "D0"), ("D1G", "D1"), ("D2G", "D2"),
    ("D1", "R3"), ("D2G", "R3"), ("D1", "C_plateau"),
    # wave 2 -- the resolution row
    ("R1", "C_plain"), ("R2", "R1"), ("R3", "R1"), ("R3", "R2"),
    ("R4", "R1"), ("R5", "R3"), ("R6", "P_postblock"),
    ("R3", "C_plain"), ("R4", "C_plain"), ("R3", "P_postblock"),
    # the effect this whole ablation is trying to explain
    ("C_plateau", "C_plain"),
    # test 1 -- placement
    ("P_postbn", "C_plateau"), ("P_postblock", "C_plateau"),
    ("P_postbn", "C_plain"), ("P_postblock", "C_plain"),
    # test 2 -- which sites
    ("M_predown", "C_plain"), ("M_nodown", "C_plain"),
    ("M_predown", "C_plateau"), ("M_nodown", "C_plateau"),
    # tests 3+4 -- constant sigma (architecture, not continuation)
    ("K_const030", "C_plain"), ("K_const050", "C_plain"),
    ("K_const080", "C_plain"), ("K_const100", "C_plain"),
    ("K_const050", "C_plateau"),
    # test 5 -- a real fixed anti-aliasing prefilter
    ("B_blurpool", "C_plain"), ("B_blurpool", "C_plateau"),
    ("B_blurpool_plateau", "C_plateau"), ("B_blurpool_plateau", "B_blurpool"),
]


def main():
    arms = collect()
    missing = [c["id"] for c in build_configs() if c["id"] not in arms]
    cmps = [c for c in (compare(arms, a, b) for a, b in COMPARISONS) if c]

    out = {"n_arms": len(arms), "missing": missing, "seeds": 1,
           "arms": arms, "comparisons": cmps,
           "caveats": [
               "one seed: signs and magnitudes only, no significance, no SD",
               "constant-sigma arms are read on the CURRENT path (primary_path)",
               "not comparable to results/campaign_results.json (different init weights)",
           ]}
    dest = ROOT / "results" / "ablation_aa_results.json"
    dest.write_text(json.dumps(out, indent=2))

    order = ["C_plain", "C_plateau", "P_postbn", "P_postblock", "M_predown",
             "M_nodown", "K_const030", "K_const050", "K_const080", "K_const100",
             "B_blurpool", "B_blurpool_plateau",
             "R1", "R2", "R3", "R4", "R5", "R6",
             "D0", "D0G", "D1", "D1G", "D2", "D2G"]
    print("%-20s %-8s %2s %7s %7s %8s %9s  %-13s %s" %
          ("arm", "path", "n", "acc", "sd", "probeCE", "shiftCons",
           "alias b3/b6", "wall"))
    base = arms.get("C_plain", {}).get("acc")
    for cid in order:
        a = arms.get(cid)
        if not a:
            print("%-20s  MANQUANT" % cid); continue
        al = a["alias"]
        print("%-20s %-8s %2d %7.4f %7s %8.3f %9.4f  %.3f/%.3f  %4.0fs  %+.2f pp"
              % (cid, a["primary_path"], a["n_seeds"], a["acc"],
                 "-" if a["acc_sd"] is None else "%.4f" % a["acc_sd"],
                 a["probe_ce"], a["shift_consistency"] or float("nan"),
                 al.get("blocks3") or float("nan"), al.get("blocks6") or float("nan"),
                 a["wall_s"], (a["acc"] - base) * 100 if base else float("nan")))
    print("\ncomparaisons appariées (points de pourcentage) :")
    for c in cmps:
        print("  %-22s - %-18s = %+6.2f pp  (%d graine%s%s)"
              % (c["a"], c["b"], c["d_acc_pp"], c["n_seeds"],
                 "s" if c["n_seeds"] > 1 else "",
                 "" if c["all_same_sign"] is None
                 else (", meme signe" if c["all_same_sign"] else ", SIGNE VARIABLE")))
    print("\nécrit -> %s" % dest)
    if missing:
        print("MANQUANT:", missing)


if __name__ == "__main__":
    main()

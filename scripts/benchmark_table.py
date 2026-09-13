"""Emit the complete benchmark table: every trained configuration, with the
conditions it ran under, plus the pilots that were never part of a batch.

Conditions are read out of each cell's own ``summary.json`` -- not retyped --
so the table cannot drift from what was actually run.  Writes
``docs/BENCHMARK_TABLE.md``.  Reads only; no training.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gather_all_methods import SCRATCH, main as gather

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "BENCHMARK_TABLE.md"

SOURCES = {
    "campaign": "results/kaggle_outputs/campaign-j*/*/*/summary.json",
    "resbench": "results/kaggle_outputs/resbench-j*/*/*/summary.json",
    "unified": "results/kaggle_outputs/unified-j*/*/*/summary.json",
    "ablation": str(SCRATCH / "ablation_data/results/kaggle_outputs/abl*-j*/*/*/summary.json"),
    "adaptive": str(SCRATCH / "aubin/results/kaggle_outputs/*/*/*/summary.json"),
}

#: batch -> the pinned asset set it used.  Arms sharing a set are seed-paired.
ASSETS = {"campaign": "A", "resbench": "A", "adaptive": "A",
          "ablation": "B", "unified": "C"}

PLACE = {"conv_out": "conv output", "post_block": "post-ReLU",
         "post_bn": "post-BatchNorm"}
#: ``input_bilinear`` with an R32 schedule is not a reduction -- it is the
#: default no-op path.  Only spell it out when the schedule actually moves.
REDUCTION = {"input_bilinear": "input image (bilinear)",
             "input_max": "input image (max-pool)",
             "stem_max": "first conv layer (max-pool)",
             "stem_bilinear": "first conv layer (bilinear)",
             "block0_max": "block 0 (max-pool)",
             "block1_max": "block 1 (max-pool)",
             "block2_max": "block 2 (max-pool)"}
OPERATOR_NAME = {"max": "max-pool", "bilinear": "bilinear",
                 "maxblur": "max-pool + anti-alias", "softpool": "softpool",
                 "l2": "least-squares", "hminus1": "smoothness-optimal",
                 "perceptual": "perceptual (SSIM-style)", "none": "none"}
WHERE_NAME = {"input": "the input image", "stem": "the first conv layer",
              "D0": "block 0", "D1": "block 1", "D2": "block 2"}


def conditions():
    """(batch, id, epochs) -> run conditions, from one representative cell."""
    out = {}
    for batch, pat in SOURCES.items():
        for f in glob.glob(pat):
            try:
                s = json.loads(Path(f).read_text(encoding="utf-8"))
            except Exception:
                continue
            # same arm-name recovery gather_all_methods uses: Aubin's cells
            # spell the seed with one underscore, ours with two.
            cell = str(s.get("cell_id") or s.get("label") or "")
            cid = (cell.rpartition("__seed")[0]
                   or cell.rsplit("_seed", 1)[0]
                   or s.get("id"))
            if not cid:
                continue
            key = (batch, cid, s.get("epochs", 30))
            if key in out:
                continue
            ctrl = s.get("controller") or {}
            res = ctrl.get("resolution_by_epoch") or []
            if not res:
                path = "32"
            elif len(set(res)) == 1:
                path = str(res[0])
            else:
                seen, seq = set(), []
                for r in res:
                    if not seq or seq[-1] != r:
                        seq.append(r)
                path = "-".join(str(r) for r in seq)
            out[key] = {
                "operator": s.get("operator") or ctrl.get("operator") or "none",
                "placement": s.get("placement") or ctrl.get("placement") or "-",
                "mask": s.get("mask") or ctrl.get("mask") or "-",
                "n_sites": ctrl.get("n_sites") or ctrl.get("n_positions") or "-",
                "location": s.get("location") or ctrl.get("location"),
                "reduction": s.get("reduction") or ctrl.get("reduction") or "none",
                "res_path": path,
                "constant": bool(s.get("constant") or ctrl.get("constant_level")),
                "profile": bool(s.get("sigma_profile") or ctrl.get("sigma_profile")),
                "adaptive": bool(s.get("adaptive") or s.get("trigger_kind")),
                "n_train": s.get("n_train"), "n_test": s.get("n_test"),
                "updates": s.get("updates"),
            }
    return out


def describe(batch, c):
    """(filter, placement, sites, reduction) as the *reader* means them.

    ``operator`` does not mean the same thing in every batch: in resbench it is
    the resolution operator and there is no filter at all, while elsewhere it
    names the activation filter.  Reading it uniformly mislabels 28 rows.
    """
    if batch == "resbench":
        op = OPERATOR_NAME.get(c.get("operator", ""), c.get("operator", "?"))
        loc = c.get("location", "?")
        if op == "none":
            return "-", "-", "-", "-"
        return "none (disabled)", "-", "-", "%s at %s" % (op, WHERE_NAME.get(loc, loc))

    if not c:
        return "not recorded", "-", "-", "-"
    filt = c.get("operator") or "none"
    if filt == "gaussian":
        filt = ("constant sigma" if c.get("constant") else
                "Gaussian, per-layer profile" if c.get("profile") else
                "Gaussian, annealed")
    elif filt in ("none", None):
        filt = "-"
    if c.get("adaptive"):
        filt = (filt if filt != "-" else "Gaussian") + ", data-triggered"
    # campaign and adaptive predate the placement experiment: every filter
    # there sits at the conv output, and their summaries do not record it.
    place = c.get("placement") or ("conv_out" if batch in ("campaign", "adaptive")
                                   else "-")
    place = "-" if filt == "-" else PLACE.get(place, place)
    mask = c.get("mask", "-")
    sites = "-" if filt == "-" else "%s%s" % (
        c.get("n_sites", "-"),
        "" if mask in ("all19", "-", None) else " (%s)" % mask)
    red = c.get("reduction", "none")
    red = REDUCTION.get(red, red)
    return filt, place, sites, (red if red not in ("none", None) else "-")


def main():
    rows = gather()
    cond = conditions()
    by_batch = defaultdict(list)
    for k, v in rows.items():
        by_batch[v["batch"]].append((k, v))

    lines = []
    w = lines.append
    w("# Complete benchmark table")
    w("")
    w("Generated by `scripts/benchmark_table.py` from the cells' own")
    w("`summary.json` files. Every row is a trained configuration; accuracy is the")
    w("**final-epoch test accuracy on the current path**, mean +/- sample SD over")
    w("the seeds listed. No best-epoch selection anywhere.")
    w("")
    w("## Shared conditions")
    w("")
    w("Unless a row says otherwise, every trained arm below ran under exactly this")
    w("setup:")
    w("")
    w("| | |")
    w("|---|---|")
    w("| Dataset | CIFAR-10, official split: **50,000 train / 10,000 test** |")
    w("| Augmentation | **none** (no crop, no flip) |")
    w("| Model | ResNet-20 + BatchNorm, widths 16/32/64, option-A shortcuts, 269,722 params |")
    w("| Budget | 30 epochs = 11,730 updates, 391 updates/epoch |")
    w("| Optimiser | SGD, LR 0.005, momentum 0.9, weight decay 5e-4 |")
    w("| Schedule | 60 warmup updates, then cosine, indexed by global update |")
    w("| Batch | effective 128, as microbatches of 32 weighted by example count |")
    w("| Normalisation | statistics computed on the full 50,000 train set |")
    w("| Evaluation | full 10,000-image test set; train probe = fixed held-in subset |")
    w("| Hardware | NVIDIA T4 (Kaggle), 4 accounts / 8 GPUs |")
    w("| Endpoint | all operators bypassed at 32x32; verified **bitwise** identical to plain ResNet-20 |")
    w("")
    w("**Pairing.** Arms sharing a pinned asset set (initial weights, BN buffers,")
    w("probe indices, per-epoch permutations) are seed-paired, so their differences")
    w("are meaningful cell by cell. Three sets exist, each verified by sha256:")
    w("")
    w("| set | batches | plain baseline |")
    w("|---|---|---|")
    w("| **A** | campaign, resbench, adaptive | 75.62 / 75.53 / 74.42 % |")
    w("| **B** | ablation (Idriss) | 75.06 % |")
    w("| **C** | unified | 75.43 % |")
    w("")
    w("Comparisons *across* sets carry a batch offset of up to ~0.6 pp.")
    w("")
    w("**Column meanings.** *filter* = what is applied to activations and how its")
    w("strength moves; *placement* = where in each block; *sites* = how many")
    w("insertion points; *reduction* = what spatial reduction is applied and where;")
    w("*resolution* = the internal resolution schedule across the 30 epochs.")
    w("")

    titles = {
        "campaign": ("campaign -- 21 configurations, 3 seeds",
                     "Sigma schedules x reduction site x insertion mask. The "
                     "first full-data batch."),
        "resbench": ("resbench -- 28 configurations, 3 seeds",
                     "Resolution reduction **only**, all internal Gaussian "
                     "disabled: 7 operators x 5 sites x 4 schedules."),
        "ablation": ("ablation (Idriss) -- 32 configurations, 1-3 seeds",
                     "Blur placement, insertion masks, constant-sigma controls "
                     "and per-layer sigma profiles."),
        "adaptive": ("adaptive (Aubin) -- 14 configurations, 1-6 seeds",
                     "Data-triggered schedules and dwell-time allocation. The "
                     "120-epoch arms are excluded: four times the budget is not "
                     "a method effect."),
        "unified": ("unified -- 16 configurations, 3 seeds",
                    "Every promising method in **one** batch, one asset set, "
                    "evaluated every epoch. The only internally paired set."),
    }

    for batch in ("unified", "ablation", "resbench", "campaign", "adaptive"):
        items = sorted(by_batch[batch], key=lambda kv: -kv[1]["acc_mean"])
        if not items:
            continue
        title, blurb = titles[batch]
        w("## %s" % title)
        w("")
        w("%s Asset set **%s**." % (blurb, ASSETS[batch]))
        w("")
        w("| acc % | SD | seeds | method | filter | placement | sites | reduction | resolution |")
        w("|---:|---:|---:|---|---|---|---:|---|---|")
        for k, v in items:
            cid = k.split("/", 1)[1].split("@")[0]
            c = cond.get((batch, cid, v["epochs"]), {})
            filt, place, sites, red = describe(batch, c)
            if red != "-" and c.get("res_path") == "32":
                red = "-"          # no-op default path, not a reduction
            w("| %.2f | %.2f | %d | %s | %s | %s | %s | %s | %s |"
              % (100 * v["acc_mean"], 100 * v["acc_sd"], v["n_seeds"],
                 v["label"], filt, place, sites, red, c.get("res_path", "32")))
        w("")

    w("## Pilots and preview-only work (different conditions)")
    w("")
    w("These predate the batches and do **not** share their setup -- read the")
    w("conditions column before comparing any of these numbers to the tables above.")
    w("")
    w("| study | dataset / budget | seeds | condition | result |")
    w("|---|---|---|---|---|")
    for r in [
        ("Fixed input Gaussian", "CIFAR-10, 45k/5k val split", "1",
         "Blur the **image**, fixed sigma; ResNet-20 GroupNorm",
         "**-3.45 pp**. Input-space intervention hurts; thread closed"),
        ("Warm starts", "10,000 images, 1,200 updates", "3 paired",
         "Transfer weights across sigma levels", "No usable gain"),
        ("TV-L2 / TV-Hminus1 budgets", "preview only, **never trained**", "-",
         "Hard relative TV budget via Chambolle-Pock (PDHG)",
         "Hminus1 retains more contrast at equal TV; 1,113 s / 2,222 s per preview -- too slow to train"),
        ("Wavelet previews", "preview only, **never trained**", "-",
         "Undecimated 2-level shrinkage: haar, db2, sym4, coif1",
         "Tight frame verified; fused path 2.0-2.1x forward"),
        ("ResNet-20+BN Gaussian pilot", "10,000 images, 2,400 updates", "1",
         "Feature-space Gaussian, 19 sites, sigma 1.00 -> 0.30",
         "**+4.26 pp**. The pivot: feature space works where input space did not"),
        ("db2 wavelet continuation", "10,000 images, 2,400 updates", "1",
         "db2 shrinkage replacing the Gaussian at the same 19 sites",
         "**+0.94 pp**, sign flips at update 1200; 4-5x weaker. **Closed**"),
        ("Progressive resolution", "50,000 / 10,000, 30 epochs", "1",
         "Input resized 16 -> 24 -> 32 before normalisation, bilinear antialias",
         "**+4.47 pp** and 6.4 % *faster*. Largest single effect found"),
    ]:
        w("| %s | %s | %s | %s | %s |" % r)
    w("")
    w("The 24 % reduction in convolution work did **not** become wall time: at")
    w("16x16 with 16 channels a T4 is far from saturated (per-update 0.0328 s at")
    w("r=16 against 0.0324 s at r=32). Memory does scale, 35 -> 72 MiB.")
    w("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote %s -- %d configurations"
          % (OUT, sum(len(v) for v in by_batch.values())))


if __name__ == "__main__":
    main()

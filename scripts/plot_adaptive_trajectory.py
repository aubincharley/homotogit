"""The realised sigma path of the adaptive arm, against the fixed profiles.

The primary question this run asks is not "is the adaptive arm more accurate" --
one seed cannot answer that -- but **what depth profile does the controller
discover when it starts uniform**.  So the left panel is the realised
`sigma_l(e)` per stage, and the right panel is the same path normalised to its
own peak at each epoch, which is directly comparable to the fixed `c_l` profiles
(rho=1 is flat at 1; rho=2 is 0.25/0.5/1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# site 0 is the stem and 1..6 are stage 1 (they share a spatial width)
GROUPS = [("stem + stage 1", 0, "#1b9e77"),
          ("stage 2", 7, "#d95f02"),
          ("stage 3", 13, "#7570b3")]


def main(out_dir="results/per_layer_cpu_v2"):
    out = Path(out_dir)
    table = json.loads((out / "adaptive" / "realised_table.json").read_text())
    summary = json.loads((out / "adaptive" / "summary.json").read_text())
    epochs = list(range(len(table)))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for label, site, c in GROUPS:
        axes[0].plot(epochs, [r[site] for r in table], "-o", ms=4, color=c,
                     label="adaptive — %s" % label)

    # the two fixed profiles, for reference, on the same shared schedule G(e)
    ref = json.loads((out / "summaries.json").read_text())
    for arm, style in (("rho1", ":"), ("rho2", "--")):
        row = next((s for s in ref if s["arm"] == arm), None)
        if row is None:
            continue
        tab = row["controller"]["level_table"]
        for label, site, c in GROUPS:
            axes[0].plot(range(len(tab)), [r[site] for r in tab], style, lw=1.0,
                         alpha=0.5, color=c)
    axes[0].plot([], [], "--", color="#888888", label=r"fixed $\rho$=2 (dashed)")
    axes[0].plot([], [], ":", color="#888888", label=r"fixed $\rho$=1 (dotted)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel(r"$\sigma_\ell$")
    axes[0].set_title(r"Realised $\sigma_\ell(e)$ — starts uniform at 1.0")
    axes[0].legend(fontsize=7.5)
    axes[0].grid(alpha=0.25)

    # normalised shape: what profile did it actually discover?
    for label, site, c in GROUPS:
        shape = []
        for r in table:
            peak = max(r) if max(r) > 0 else 0.0
            shape.append(r[site] / peak if peak > 0 else float("nan"))
        axes[1].plot(epochs, shape, "-o", ms=4, color=c, label=label)
    for lvl, txt in ((1.0, r"$\rho$=2 deep"), (0.5, r"$\rho$=2 stage2"),
                     (0.25, r"$\rho$=2 shallow")):
        axes[1].axhline(lvl, color="#d62728", ls="--", lw=0.9, alpha=0.55)
        axes[1].annotate(txt, (0.02, lvl), xycoords=("axes fraction", "data"),
                         fontsize=7, color="#d62728", va="bottom")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel(r"$\sigma_\ell\,/\,\max_\ell \sigma_\ell$")
    axes[1].set_title("Discovered profile shape vs the fixed $\\rho$=2 targets")
    axes[1].legend(fontsize=8, loc="center right")
    axes[1].grid(alpha=0.25)

    fired = summary.get("deadline_fired")
    fig.suptitle("Adaptive predictor-corrector: realised path (1 seed, %d steps, "
                 "deadline_fired=%s)" % (len(summary.get("sigma_steps", [])), fired),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = out / "adaptive_trajectory.png"
    fig.savefig(png, dpi=150)
    print("wrote", png, png.stat().st_size, "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:])

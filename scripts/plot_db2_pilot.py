"""Pilot figure: plain / Gaussian / db2 on ResNet-20 + BatchNorm, seed 0.

Active-filter curves are primary; the exact-bypass evaluation is a thin dashed
diagnostic (for a BatchNorm model it is misleading while filters are active,
because the running statistics were accumulated with the filter in place).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NEW = Path("results/kaggle_outputs/db2-pilot-r20bn-20260908-191741/"
           "db2_pilot_r20bn_20260908-191900")
OLD = Path("results/kaggle_outputs/resnet20bn-gaussian-20260908-154226/"
           "resnet20bn_gaussian_20260908-154247")
RUNS = [("plain (paired control)", NEW / "plain_r20bn_seed0", "0.35", "-"),
        ("Gaussian (separate job)", OLD / "gaussian_r20bn_seed0", "tab:blue", "-"),
        ("db2 wavelet", NEW / "db2_r20bn_seed0", "tab:red", "-")]
CK = [600, 1200, 1700, 2400]
OUT = Path("results/db2_pilot.png")


def main():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for label, d, col, ls in RUNS:
        m = json.loads((d / "metrics.json").read_text())
        k = [r["update"] for r in m]

        def series(field):
            return ([r["update"] for r in m if r.get(field) is not None],
                    [r[field] for r in m if r.get(field) is not None])

        for i, (act, byp) in enumerate((("val_acc_filtered", "val_acc"),
                                        ("val_ce_filtered", "val_ce"),
                                        ("train_probe_ce_filtered",
                                         "train_probe_ce_bypassed"))):
            xa, ya = series(act)
            ax[i].plot(xa, ya, ls, color=col, lw=2.0, label=label)
            xb, yb = series(byp)
            ax[i].plot(xb, yb, "--", color=col, lw=0.8, alpha=0.65)

    for i, t in enumerate(("validation accuracy", "validation CE",
                           "training-probe CE")):
        ax[i].set_title(t)
        ax[i].set_xlabel("optimizer update")
        ax[i].grid(alpha=0.3)
        for c in CK:
            ax[i].axvline(c, color="0.85", lw=0.7, zorder=0)
        ax[i].axvline(1700, color="0.5", lw=1.0, ls=":", zorder=0)
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].annotate("filters off\n(k >= 1700)", xy=(1700, 0.15), fontsize=7,
                   color="0.4", ha="left")
    fig.suptitle("db2 pilot -- ResNet-20 + BatchNorm, seed 0, 10k images, 2,400 updates\n"
                 "solid = active operator (primary);  thin dashed = exact bypass "
                 "(diagnostic, unreliable for BN while filters are active)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

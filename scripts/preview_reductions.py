"""input_bilinear vs input_max at 16 and 24, on the existing ten preview images."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from continuation.campaign_ops import reduce_spatial, describe_adaptive
from continuation.config import DataConfig
from continuation.data import build_dataset

IDX = [14069, 16264, 16752, 22158, 23268, 23977, 27811, 30987, 35936, 43986]
OUT = Path("results/campaign_previews")

def main():
    b = build_dataset(DataConfig(root="data", download=False, num_val=0))
    x = b.train.images[torch.as_tensor(IDX)].to(torch.float32) / 255.0
    rows = [("original 32", x)]
    for r in (24, 16):
        for how in ("input_bilinear", "input_max"):
            rows.append(("%s %d" % (how.replace("input_", ""), r),
                         reduce_spatial(x, r, how)))
    fig, ax = plt.subplots(len(rows), 10, figsize=(14, 1.5 * len(rows)))
    for i, (name, t) in enumerate(rows):
        for j in range(10):
            a = ax[i][j]
            a.imshow(t[j].permute(1, 2, 0).clamp(0, 1).numpy(), interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            if j == 0:
                a.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle("Input reduction paths (no parameter tuning; display only)", fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "reductions.png", dpi=140, bbox_inches="tight")
    (OUT / "adaptive_windows.json").write_text(json.dumps(
        {"32->24": describe_adaptive(32, 24), "32->16": describe_adaptive(32, 16)},
        indent=2))
    print("wrote", OUT / "reductions.png")

if __name__ == "__main__":
    main()

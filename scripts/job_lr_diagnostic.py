"""Phase 2: peak-LR diagnostic on the original 5,000-image pilot setting.

Everything except the peak LR is held at the pilot's values: same 5,000 images
(seed 0 subset), same initialization and minibatch order, effective batch 128,
600 updates, 30-update warmup, cosine to zero, no filtering.

This varies the peak LR only.  It does NOT test whether the short warmup caused
the original stagnation; warmup stays at 30.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.continuation_driver import run_study

CFG = {
    "name": "lr_diagnostic",
    "subset_seed": 0, "per_class": 500, "n_subset": 5000,
    "seeds": [0], "probe_seed": 0, "train_probe": 200,
    "updates": 600, "warmup": 30, "cont_end": 300,
    "effective_batch": 128, "microbatch": 32, "eval_every": 50,
    "runs": [
        {"label": "plain_lr0p02", "kind": "none", "seed": 0, "lr": 0.02},
        {"label": "plain_lr0p005", "kind": "none", "seed": 0, "lr": 0.005},
    ],
}

if __name__ == "__main__":
    run_study(CFG)

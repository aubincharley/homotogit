"""Phase 3 control: plain and Gaussian at peak LR 0.02, seed 0.

Frozen protocol identical to the 0.005 pair (10,000 images, 5,000 val, 1,200
updates, warmup 60, cosine, effective batch 128 = 32 x 4, same subset / init /
minibatch order, same sigma_k = max(1 - k/600, 0)).  Only the peak LR differs.
The two runs go to the two GPUs independently.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._study_common import BASE
from scripts.continuation_driver import run_study

CFG = {"name": "lr_control_002", **BASE, "seeds": [0],
       "runs": [{"label": "plain_seed0_lr0p02", "kind": "none", "seed": 0, "lr": 0.02},
                {"label": "gaussian_seed0_lr0p02", "kind": "gaussian", "seed": 0, "lr": 0.02}]}

if __name__ == "__main__":
    run_study(CFG)

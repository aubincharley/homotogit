"""Paired plain vs Gaussian on ResNet-20 + BatchNorm with corrected initialization.

Seed 0, 10,000 training images, 5,000 disjoint validation images, no augmentation.
2,400 optimizer updates, effective batch 128 as 4 physical microbatches of 32
(BatchNorm sees each microbatch of 32).  SGD peak LR 0.005, momentum 0.9,
wd 5e-4, no Nesterov; 60-update warmup then cosine to the end of 2,400.

Piecewise-constant sigma, held flat within each plateau; 1,700 filtered updates
then 700 unfiltered with exact bypass.  The plain arm bypasses throughout.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._study_common import BASE
from scripts.continuation_driver import run_study

SIGMA_PIECEWISE = [[0, 1.00], [250, 0.85], [500, 0.70], [750, 0.60],
                   [1000, 0.50], [1250, 0.40], [1500, 0.30], [1700, 0.00]]

CFG = {**BASE,
       "name": "resnet20bn_gaussian",
       "arch": "resnet20_bn_cifar",
       "seeds": [0],
       "updates": 2400,
       "warmup": 60,
       "cont_end": 1700,                 # informational; the piecewise table rules
       "sigma_piecewise": SIGMA_PIECEWISE,
       "eval_every": 200,
       "checkpoint_updates": [600, 1200, 1700, 2400],
       "runs": [{"label": "plain_r20bn_seed0", "kind": "none", "seed": 0, "lr": 0.005},
                {"label": "gaussian_r20bn_seed0", "kind": "gaussian", "seed": 0, "lr": 0.005}]}

if __name__ == "__main__":
    run_study(CFG)

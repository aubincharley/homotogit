"""Full-data ResNet-20 + BatchNorm campaign: 9 runs (3 seeds x 3 arms), 30 epochs.

Official CIFAR-10 split: all 50,000 training images, 10,000 test images (held-out
curves are labelled "test").  No augmentation; normalization statistics computed
once from the full training split.  Effective batch 128 as microbatches of 32,
keeping the partial final batch of each epoch and weighting every microbatch by
its actual example count.  SGD peak LR 0.005, momentum 0.9, wd 5e-4, no Nesterov,
60 warmup updates then cosine over the whole budget.

Arms (e = zero-based epoch):
  plain       - no filtering
  plateau     - 1.00/0.85/0.70/0.60/0.50/0.40/0.30 in 3-epoch blocks, bypass from e=21
  geometric   - sigma = 0.9**e for e<=20, bypass from e=21  ("compressed geometric":
                the paper's initial sigma and factor with a one-epoch interval and an
                explicit final bypass -- NOT an exact CBS reproduction)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.continuation_driver import run_study

EPOCHS, BYPASS_FROM = 30, 21
PLATEAU = [1.00, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30]
plateau_sigmas = [PLATEAU[min(e // 3, len(PLATEAU) - 1)] if e < BYPASS_FROM else 0.0
                  for e in range(EPOCHS)]
geometric_sigmas = [0.9 ** e if e < BYPASS_FROM else 0.0 for e in range(EPOCHS)]

SEEDS = [0, 1, 2]
ARMS = [("plain", "none", None),
        ("plateau", "gaussian", plateau_sigmas),
        ("geometric", "gaussian", geometric_sigmas)]

CFG = {
    "name": "fulldata_r20bn",
    "arch": "resnet20_bn_cifar",
    "subset_seed": 0, "per_class": 5000, "n_subset": 50000,
    "num_val": 0,                       # no held-out split: train on all 50,000
    "eval_split": "test",               # official 10,000-image test set
    "seeds": SEEDS, "probe_seed": 0, "train_probe": 500,
    "epochs": EPOCHS, "warmup": 60,
    "effective_batch": 128, "microbatch": 32,
    "eval_every_epochs": 2, "eval_extra_epochs": [BYPASS_FROM],
    "checkpoint_epochs": [BYPASS_FROM, EPOCHS],
    "cont_end": 1,                      # unused in epoch mode
    "runs": [{"label": "%s_seed%d" % (name, s), "kind": kind, "seed": s, "lr": 0.005,
              **({"sigma_by_epoch": sig} if sig else {})}
             for s in SEEDS for name, kind, sig in ARMS],
}

if __name__ == "__main__":
    run_study(CFG)

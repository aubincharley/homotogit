"""Per-layer sigma profiles and the adaptive controller, full CIFAR-10, T4.

Scales the laptop pilot up to the protocol the rest of the repository uses, so
the result can be read against the campaign's **measured** single-seed noise
floor of ~0.1 pp instead of an unmeasured one: official 50,000 / 10,000 split,
30 epochs, SGD peak LR 0.005 / momentum 0.9 / wd 5e-4, 60-update warmup then
cosine over the whole budget, no augmentation, filters bypassed from epoch 21 so
every arm ends on nine complete epochs at the exact target objective.

Deviation from the campaign, recorded rather than hidden: batch 128 is applied
directly rather than as four microbatches of 32.  Gradient accumulation was
verified exact (3.97e-16 in float64), so this changes only BatchNorm's batch
statistics, which the campaign's microbatching also changed.  It is held fixed
across every arm here, so the paired comparison is unaffected.

Five arms per seed, all sharing initial weights, subset, test subset and every
per-epoch permutation, verified by sha256 before a single update runs:

    plain      no filtering
    rho1       uniform sigma at all 19 sites (the CBS-style schedule)
    rho0.5     c = (1, 1/2, 1/4)  -- constant physical scale
    rho2       c = (1/4, 1/2, 1)  -- blur deep; the laptop pilot's winner
    adaptive   predictor-corrector; starts uniform and discovers its own profile

The pilot ran one seed at 6,000 images with no measured noise floor, so it
established a direction, not a result.  Three seeds here is descriptive, not
inferential -- the repo's errata C-12/C-13 are explicit that a handful of runs
defines no distribution.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEEDS = (0, 1, 2)
# rho0.5 / rho2 are unaffected by the controller fix and are already
# measured in per-layer-sigma-20260910-094757; re-running plain and rho1
# identically also yields a GPU run-to-run noise floor at this scale.
ARMS = "plain,rho1,adaptive"
OUT = Path(os.environ.get("STUDY_OUT", "/kaggle/working")) / "per_layer_gpu"

COMMON = [
    "--n-train", "50000", "--n-test", "10000",
    "--epochs", "30", "--filtered-epochs", "21",
    "--terminal-epochs", "9", "--ramp-stages", "2",
    "--batch", "128", "--lr", "0.005", "--warmup", "60",
    "--probe-size", "1024", "--eval-batch", "1000",
    "--arms", ARMS,
]


def main():
    from scripts.study_per_layer_cpu import main as study
    t0 = time.perf_counter()
    for seed in SEEDS:
        out = OUT / ("seed%d" % seed)
        print("=" * 70, flush=True)
        print("SEED %d -> %s   [%.0fs elapsed]" % (seed, out, time.perf_counter() - t0),
              flush=True)
        print("=" * 70, flush=True)
        study(COMMON + ["--seed", str(seed), "--out", str(out)])
    print("all seeds done in %.0fs" % (time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
    main()

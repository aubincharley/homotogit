"""Amplitude sweep: is sigma0 = 1.0 the right ceiling, or just an inherited default?

EXP-012 showed the *shape* of the schedule is a flat axis -- a uniform sigma beats
every depth profile tested, and an adaptive controller matches but does not beat
it.  What has never been varied is the **amplitude**.  ``SiteController`` hardcoded
``GaussianSmoothing(sigma_max=1.0)``, and every profile normalises to
``max c_l = 1``, so sigma0 = 1.0 has been a hard ceiling on the whole family since
the beginning -- inherited from the reference implementation rather than chosen.

Six arms per seed, identical apart from the amplitude multiplying one shared
plateau schedule:

    plain     no filtering
    s0_0.25   quarter amplitude            radius 1,  3 taps
    s0_0.5    half amplitude               radius 2,  5 taps
    rho1      sigma0 = 1.0, the incumbent  radius 4,  9 taps
    s0_1.5    one and a half               radius 6, 13 taps
    s0_2      double                       radius 8, 17 taps

``radius = ceil(4 * sigma_max)`` scales with the amplitude, so every arm keeps the
same **relative** truncation of +-4 sigma; the support in pixels differs between
arms but the operator quality is matched.  At sigma0 = 2 the radius reaches 8 on
the 8x8 stage-3 maps, which is exactly where PyTorch's native reflect padding
refuses and the explicit whole-sample reflection gather takes over -- verified
finite in forward and backward before launching.

Protocol is EXP-012's, unchanged, so results are readable against its measured
~0.5 pp noise floor: 50k/10k, 30 epochs, 11,730 updates, LR 0.005, warmup 60 then
cosine, no augmentation, filters bypassed from epoch 21, seeds 0/1/2 paired by
sha256 on initial weights, subsets and every per-epoch permutation.

What this cannot answer: amplitude and effective schedule are not separable.  A
larger sigma0 also means the schedule spends longer at widths the smaller arms
never reach, so a difference here is not attributable to amplitude alone.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEEDS = (0, 1, 2)
ARMS = "plain,s0_0.25,s0_0.5,rho1,s0_1.5,s0_2"
OUT = Path(os.environ.get("STUDY_OUT", "/kaggle/working")) / "sigma0_sweep"

COMMON = [
    "--n-train", "50000", "--n-test", "10000",
    "--epochs", "30", "--filtered-epochs", "21",
    "--terminal-epochs", "9", "--ramp-stages", "2",
    "--batch", "128", "--lr", "0.005", "--warmup", "60",
    "--probe-size", "1024", "--eval-batch", "1000",
    "--arms", ARMS,
]


def main():
    from scripts import study_per_layer_cpu as study_mod
    t0 = time.perf_counter()
    for seed in SEEDS:
        out = OUT / ("seed%d" % seed)
        print("=" * 70, flush=True)
        print("SEED %d -> %s   [%.0fs]" % (seed, out, time.perf_counter() - t0), flush=True)
        print("=" * 70, flush=True)
        study_mod.main(COMMON + ["--seed", str(seed), "--out", str(out)])
    print("all seeds done in %.0fs" % (time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
    main()

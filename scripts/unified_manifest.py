"""One batch containing every promising method, so their curves finally share an axis.

Everything here trains from the **same pinned initial weights, BN buffers, probe
indices and per-epoch permutations**, and evaluates **every epoch** rather than
every second epoch.  That is the whole point: the arms we care about currently
live in three different batches, and the ablation batch carries a ~0.56 pp offset
that makes a single loss plot dishonest.

Three groups.

**A. References and best-known arms** -- the baseline, the two blur placements,
and the resolution-only winners from the resolution benchmark.

**B. Idriss's best, replayed here** -- his top two arms reproduced with his own
verified operator code (``continuation/ablation_ops.py``, ported unchanged), so
any difference from his numbers is the batch, not the implementation.

**C. New crossings** -- the resolution-only winners crossed with the Gaussian at
the placements that worked for him.  This is the gap in the current evidence:
we know max-pool reduction at D0/D1/D2 works alone, and we know post-ReLU blur
works alone, but only ``D2 + post-ReLU blur`` has been run as a pair.

Each cell is a dict consumed by ``continuation.ablation_ops.build_from_cell``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEEDS = (0, 1, 2)

#: Idriss's receptive-field profile, 10 post-block positions (his P_A4 arm).
PROFILE_RF = [0.09375, 0.21875, 0.34375, 0.46875, 0.65625, 0.90625,
              1.0, 1.0, 1.0, 1.0]
#: sqrt(map size) profile, his P_A2 arm.
PROFILE_SQRT = [1.0, 1.0, 1.0, 1.0, 0.70710678, 0.70710678, 0.70710678,
                0.5, 0.5, 0.5]


def _cell(cid, label, group, operator, levels, resolution, reduction,
          placement="conv_out", sigma_profile=None, priority=1):
    return {"id": cid, "label": label, "group": group,
            "operator": operator, "levels": levels, "resolution": resolution,
            "reduction": reduction, "placement": placement,
            "mask": "all19", "blurpool_sigma": None, "constant": False,
            "primary_path": "target", "sigma_profile": sigma_profile,
            "relative_reduction": False, "diagnostics": False,
            "controller_builder": "continuation.ablation_ops.build_from_cell",
            "priority": priority}


def build_configs() -> list:
    c = []

    # -- A. references and best-known -------------------------------------
    c.append(_cell("plain", "Baseline: no blur, full 32x32 throughout", "A",
                   "none", "none", "R32", "input_bilinear"))
    c.append(_cell("blur_conv", "Blur every conv layer, annealed to zero", "A",
                   "gaussian", "plateau", "R32", "input_bilinear", "conv_out"))
    c.append(_cell("blur_relu", "Blur after every ReLU, annealed to zero", "A",
                   "gaussian", "plateau", "R32", "input_bilinear", "post_block"))
    c.append(_cell("shrink_input", "Shrink the input image, 16 to 24 to 32", "A",
                   "none", "none", "Rprog", "input_bilinear"))
    for k in (0, 1, 2):
        c.append(_cell("shrink_b%d" % k,
                       "Shrink block %d with max-pool, 16 to 24 to 32" % k, "A",
                       "none", "none", "Rprog", "block%d_max" % k))

    # -- B. Idriss's best two, replayed on these assets --------------------
    c.append(_cell("shrink_b2_relu",
                   "Shrink block 2, plus blur after every ReLU", "B",
                   "gaussian", "plateau", "Rprog", "block2_max", "post_block"))
    c.append(_cell("shrink_b2_relu_rf",
                   "Shrink block 2, plus blur scaled by receptive field "
                   "after every ReLU", "B",
                   "gaussian", "plateau", "Rprog", "block2_max", "post_block",
                   PROFILE_RF))

    # -- C. new crossings: resolution winners x blur placement -------------
    for k in (0, 1):
        c.append(_cell("shrink_b%d_relu" % k,
                       "Shrink block %d, plus blur after every ReLU" % k, "C",
                       "gaussian", "plateau", "Rprog", "block%d_max" % k,
                       "post_block"))
        c.append(_cell("shrink_b%d_relu_rf" % k,
                       "Shrink block %d, plus blur scaled by receptive field "
                       "after every ReLU" % k, "C",
                       "gaussian", "plateau", "Rprog", "block%d_max" % k,
                       "post_block", PROFILE_RF))
    # placement control: the same reduction with the blur at the conv output
    c.append(_cell("shrink_b1_conv",
                   "Shrink block 1, plus blur every conv layer", "C",
                   "gaussian", "plateau", "Rprog", "block1_max", "conv_out"))
    # profile control: sqrt(map size) instead of receptive field
    c.append(_cell("shrink_b1_relu_sqrt",
                   "Shrink block 1, plus blur scaled by sqrt of map size "
                   "after every ReLU", "C",
                   "gaussian", "plateau", "Rprog", "block1_max", "post_block",
                   PROFILE_SQRT, priority=2))
    # input-side reduction crossed with the better blur placement
    c.append(_cell("shrink_input_relu",
                   "Shrink the input image, plus blur after every ReLU", "C",
                   "gaussian", "plateau", "Rprog", "input_bilinear",
                   "post_block"))
    return c


def build_cells(configs=None) -> list:
    configs = configs or build_configs()
    return [{**cfg, "seed": s, "cell_id": "%s__seed%d" % (cfg["id"], s)}
            for cfg in configs for s in SEEDS]


if __name__ == "__main__":
    cfgs = build_configs()
    cells = build_cells(cfgs)
    print("configurations: %d (unique %d)" % (len(cfgs), len({c['id'] for c in cfgs})))
    print("cells:          %d" % len(cells))
    from collections import Counter
    print("by group:", dict(Counter(c["group"] for c in cfgs)))
    for c in cfgs:
        print("  [%s] %-22s %s" % (c["group"], c["id"], c["label"]))

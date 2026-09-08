"""Frozen protocol shared by the plain / Gaussian / db2 studies.

Only `name` and `runs` differ between phases.  Peak LR 0.005 was selected by the
phase-2 diagnostic on final training-probe CE.
"""
PEAK_LR = 0.005

BASE = {
    "subset_seed": 0, "per_class": 1000, "n_subset": 10000,   # superset of the pilot's 5,000
    "seeds": [0, 1, 2], "probe_seed": 0, "train_probe": 500,  # balanced, 50/class
    "updates": 1200, "warmup": 60, "cont_end": 600,
    "effective_batch": 128, "microbatch": 32, "eval_every": 200,
}


def runs(kind, s0=1.0):
    return [{"label": "%s_seed%d" % (kind, s), "kind": kind, "seed": s,
             "lr": PEAK_LR, **({"s0": s0} if kind == "db2" else {})}
            for s in BASE["seeds"]]

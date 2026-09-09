"""The frozen 21-configuration manifest, reuse detection and 6-GPU scheduling.

Groups A-E give exactly 21 unique configurations; crossed with seeds 0/1/2 that
is 63 configuration-seed cells.  Cells already produced by a compatible earlier
campaign are reused; everything else is dispatched.

Reuse is decided by **artifact hashes**, never by seed number or by metrics
looking similar: a candidate must have the same initial weights and BN buffers,
the same per-epoch permutations, the same probe indices and the same protocol
fields as the campaign we are about to run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "results" / "kaggle_outputs"

EPOCHS, BYPASS_FROM, SEEDS = 30, 21, (0, 1, 2)

# -- schedules --------------------------------------------------------------

RESOLUTIONS = {
    "R32":      [32] * 30,
    "Rprog":    [16] * 6 + [24] * 6 + [32] * 18,
    "Rgentle":  [24] * 6 + [24] * 6 + [32] * 18,
    "Rreverse": [24] * 6 + [16] * 6 + [32] * 18,
}

_PLATEAU = [1.00, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30]
GAUSSIAN = {
    "Gnone":    None,
    "Gplateau": [_PLATEAU[min(e // 3, 6)] if e < BYPASS_FROM else 0.0
                 for e in range(EPOCHS)],
    "Ggeo":     [0.9 ** e if e < BYPASS_FROM else 0.0 for e in range(EPOCHS)],
    # Gmix reuses the plateau levels as alpha, then alpha = 0 from epoch 21
    "Gmix":     [_PLATEAU[min(e // 3, 6)] if e < BYPASS_FROM else 0.0
                 for e in range(EPOCHS)],
}
OPERATOR = {"Gnone": "none", "Gplateau": "gaussian", "Ggeo": "gaussian",
            "Gmix": "gmix"}


def build_configs() -> list:
    """The 21 frozen configurations, in manifest order."""
    cfgs, seen = [], set()

    def add(group, res, gauss, reduction, mask):
        cid = "%s__%s__%s__%s" % (res, gauss, reduction, mask)
        if cid in seen:
            raise ValueError("duplicate configuration %s" % cid)
        seen.add(cid)
        cfgs.append({"id": cid, "group": group, "resolution": res,
                     "gaussian": gauss, "reduction": reduction, "mask": mask,
                     "operator": OPERATOR[gauss]})

    for res in ("R32", "Rprog", "Rgentle"):                      # A: 9
        for g in ("Gnone", "Gplateau", "Ggeo"):
            add("A", res, g, "input_bilinear", "all19")
    for res in ("R32", "Rprog"):                                 # B: 2
        add("B", res, "Gmix", "input_bilinear", "all19")
    for res in ("R32", "Rprog"):                                 # C: 2
        add("C", res, "Gplateau", "input_bilinear", "early7")
    for red in ("input_max", "stem_bilinear", "stem_max"):       # D: 6
        for g in ("Gnone", "Gplateau"):
            add("D", "Rprog", g, red, "all19")
    for g in ("Gnone", "Gplateau"):                              # E: 2
        add("E", "Rreverse", g, "input_bilinear", "all19")
    return cfgs


def build_cells() -> list:
    cells = []
    for c in build_configs():
        for s in SEEDS:
            cells.append({**c, "seed": s, "cell_id": "%s__seed%d" % (c["id"], s)})
    return cells


# -- reuse ------------------------------------------------------------------

REUSE_SOURCES = {
    # historical label -> (job dir, study dir, run label, configuration id)
    "fulldata-r20bn-20260908-161221": "fulldata_r20bn_20260908-161234",
    "progres-r20bn-20260909-075846": "progres_r20bn_20260909-075919",
}

# how a historical run label maps onto a manifest configuration
HISTORICAL = {
    "plain":         ("R32", "Gnone", "input_bilinear", "all19"),
    "plateau":       ("R32", "Gplateau", "input_bilinear", "all19"),
    "geometric":     ("R32", "Ggeo", "input_bilinear", "all19"),
    "progres":       ("Rprog", "Gnone", "input_bilinear", "all19"),
    "progres_gauss": ("Rprog", "Gplateau", "input_bilinear", "all19"),
}

PROTOCOL = {"epochs": 30, "updates": 11730, "updates_per_epoch": 391,
            "warmup": 60, "effective_batch": 128, "microbatch": 32,
            "n_train": 50000, "n_test": 10000}


def _sha_array(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _sha_state(path):
    import numpy as np
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].cpu().numpy()).tobytes())
    return h.hexdigest()


def study_assets(study: Path) -> dict:
    import numpy as np
    out = {}
    npz = study / "shared_indices.npz"
    if npz.is_file():
        z = np.load(npz)
        for k in z.files:
            out[k] = _sha_array(z[k])
    for init in sorted(study.glob("init_seed*.pt")):
        out[init.stem] = _sha_state(init)
    return out


def find_reusable(reference: dict) -> dict:
    """Map cell_id -> provenance for historical cells that match ``reference``.

    ``reference`` holds the hashes the campaign will itself use.  A candidate is
    accepted only if its subset, probe, permutation and initial-weight hashes all
    agree for that seed *and* its recorded protocol matches.
    """
    found = {}
    for job, study_name in REUSE_SOURCES.items():
        study = OUTPUTS / job / study_name
        if not study.is_dir():
            continue
        assets = study_assets(study)
        for run in sorted(p for p in study.iterdir() if p.is_dir()):
            summ = run / "summary.json"
            if not summ.is_file():
                continue
            s = json.loads(summ.read_text())
            label = s.get("label", "")
            base = label.rsplit("_seed", 1)[0]
            if base not in HISTORICAL:
                continue
            seed = s.get("seed")
            res, g, red, mask = HISTORICAL[base]
            cell = "%s__%s__%s__%s__seed%d" % (res, g, red, mask, seed)

            checks = {
                "subset": assets.get("subset") == reference.get("subset"),
                "train_probe": assets.get("train_probe") == reference.get("train_probe"),
                "perm": (assets.get("perm_seed%d" % seed)
                         == reference.get("perm_seed%d" % seed)),
                "init": (assets.get("init_seed%d" % seed)
                         == reference.get("init_seed%d" % seed)),
            }
            checks.update({k: s.get(k) == v for k, v in PROTOCOL.items()
                           if k in s})
            if all(checks.values()):
                found[cell] = {"status": "reused", "job": job, "study": study_name,
                               "run_label": label, "path": str(run.relative_to(ROOT)),
                               "checks": checks,
                               "test_acc": s.get("final_test_acc"),
                               "test_ce": s.get("final_test_ce"),
                               "wall_seconds": s.get("wall_seconds")}
            else:
                found.setdefault("_rejected", []).append(
                    {"cell": cell, "run": label, "failed":
                     [k for k, v in checks.items() if not v]})
    return found


# -- scheduling -------------------------------------------------------------

def estimate_seconds(cell: dict, probe: dict | None = None) -> float:
    """Predicted wall seconds, from the timing probe when available.

    Falls back to the campaign's measured 452 s unfiltered / 674 s filtered.
    These are scheduling estimates for new operators, not guarantees.
    """
    filtered = cell["operator"] != "none"
    base = 674.0 if filtered else 452.0
    if probe:
        key = "%s__%s" % (cell["reduction"], "filtered" if filtered else "plain")
        if key in probe and probe[key].get("seconds_per_update"):
            base = probe[key]["seconds_per_update"] * PROTOCOL["updates"] * 1.10
    if cell["mask"] == "early7" and filtered:
        base *= 0.75                     # 7 of 19 sites carry the filter
    if cell["operator"] == "gmix":
        base *= 1.05                     # one extra add/scale per active site
    # progressive schedules spend 12 of 30 epochs below 32x32
    if cell["resolution"] != "R32":
        base *= 0.97
    return base


def assign(cells_pending: list, n_jobs=3, workers_per_job=2, probe=None) -> dict:
    """Longest-first assignment to (job, worker) balancing predicted finish time.

    Seeds of one configuration are spread across jobs where possible, and the
    seed->job rotation shifts per configuration so no environment is tied to a
    seed.  Within a job the two workers pull from one shared queue at run time;
    this pre-assignment only balances the per-job totals.
    """
    for c in cells_pending:
        c["est_seconds"] = estimate_seconds(c, probe)

    by_cfg = {}
    for c in cells_pending:
        by_cfg.setdefault(c["id"], []).append(c)

    load = [0.0] * n_jobs
    jobs = [[] for _ in range(n_jobs)]
    order = sorted(by_cfg.items(),
                   key=lambda kv: -sum(c["est_seconds"] for c in kv[1]))
    for i, (_cid, group) in enumerate(order):
        group = sorted(group, key=lambda c: c["seed"])
        for j, cell in enumerate(group):
            # rotate so seed s does not always land in the same environment
            pref = [(i + j + k) % n_jobs for k in range(n_jobs)]
            target = min(pref, key=lambda t: load[t])
            cell["job"] = target
            jobs[target].append(cell)
            load[target] += cell["est_seconds"]

    for j in range(n_jobs):
        jobs[j].sort(key=lambda c: -c["est_seconds"])       # longest-first queue
    return {"jobs": jobs,
            "predicted_job_seconds": load,
            "predicted_elapsed_seconds": [t / workers_per_job for t in load],
            "predicted_campaign_seconds": max(t / workers_per_job for t in load),
            "cumulative_gpu_seconds": sum(load)}


if __name__ == "__main__":
    cfgs, cells = build_configs(), build_cells()
    print("configurations: %d (unique %d)" % (len(cfgs), len({c['id'] for c in cfgs})))
    print("cells:          %d (unique %d)" % (len(cells), len({c['cell_id'] for c in cells})))
    for g in "ABCDE":
        n = len([c for c in cfgs if c["group"] == g])
        print("  group %s: %d configurations" % (g, n))

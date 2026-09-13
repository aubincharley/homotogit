"""Long-horizon convergence test: 120 epochs, two learning rates, one seed.

Six cells, ResNet-20 + BN, official 50,000 / 10,000 CIFAR-10 split, constant
resolution 32, Gaussian at all 19 sites where filtered:

    LR 0.005 (campaign protocol)      LR 0.05 (10x)
    ----------------------------      -------------
    plain      (no filtering)         plain
    Gplateau   (fixed schedule)       Gplateau
    ADAPTGAP   (transfer-gap trigger) ADAPTGAP

Why two learning rates
----------------------
At 30 epochs and LR 0.005 nothing converges: probe CE was still falling
monotonically at the final epoch (0.4150 -> 0.4112), which is why every
convergence-style trigger had nothing to detect.  Extending to 120 epochs at the
same LR only quadruples the total LR mass -- still about 6x less than the
Experiment-0 runs needed to reach train interpolation (probe acc 1.0000 by epoch
~35 at LR 0.1).  So the 0.005 leg keeps comparability with every recorded result
and the 0.05 leg is the one expected actually to converge.  Reporting both means
the convergence question is answered without discarding the comparable baseline.

Schedule scaling
----------------
The 30-epoch Gplateau is 7 levels x 3 epochs then bypass from 21 (nine terminal
epochs, 30% of the budget).  Scaled 4x: 7 levels x 12 epochs, bypass from 84,
36 terminal epochs -- the same *shape*, so dwell fraction and terminal fraction
are preserved rather than the absolute epoch counts.  ADAPTGAP's deadline scales
the same way (ramp from 72, zero by 84) and its per-stage deadline becomes 12
epochs to match one Gplateau plateau.

What this is meant to show
--------------------------
Train and test separately, over a horizon long enough to overfit.  With no
augmentation all six arms should drive train accuracy up hard; the question is
whether the continuation changes the **generalisation gap**, and whether the
gap trigger behaves differently once it has room to traverse the path instead of
being cut off by the deadline (which fired in all three 30-epoch adaptive runs).

Pairing
-------
Subset, train probe and initial weights are checked against the campaign's
pinned digests exactly.  The per-epoch permutation array is necessarily
different -- it has 120 rows rather than 30 -- so its **first 30 rows** are
checked against the campaign digest instead, which proves the same generator
stream and therefore the same minibatch order over the first 30 epochs.

Data layout
-----------
This account's CIFAR dataset stores the batch files flat rather than inside a
``cifar-10-batches-py`` directory, which is what ``data_source`` globs for.  A
symlink is created in the working directory and ``STUDY_DATA`` pointed at it, so
no 170 MB re-upload is needed.
"""
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from continuation.config import DataConfig, ModelConfig
from continuation.data import build_dataset, fixed_subset_indices
from continuation.models import build_model
from continuation.seeding import numpy_generator
from scripts.campaign_driver import log, run_job

EPOCHS, SEED = 120, 0
PER_CLASS, N_SUBSET, PROBE = 5000, 50000, 500
SCALE = EPOCHS // 30                      # 4
PLATEAU = [1.00, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30]
DWELL = 3 * SCALE                         # 12 epochs per level
BYPASS_FROM = 21 * SCALE                  # 84
RAMP_FROM = 18 * SCALE                    # 72

GPLATEAU = [PLATEAU[min(e // DWELL, 6)] if e < BYPASS_FROM else 0.0
            for e in range(EPOCHS)]
R32 = [32] * EPOCHS

EXPECTED = {
    "subset": "33236cc6bd19fa6b89e06d441d3fcd8eb37dc8540f6a4f2b627b20af10894a41",
    "train_probe": "a33645e0e4f024d8c47c0c5bdd7fbc7ea59a976090a815c58ba029908f2c86bc",
    "perm_seed0_first30": "5df32d0d922962e72fb4dd718a5eaad13f3209f75b8363aff64e57b89b4bb22f",
    "init_seed0": "6e652c49e5139e785887b9cb6b08ce5b0dee734bde6384550be6a8a3641e2c1b",
}

ADAPT = {
    "sigma_init": 1.0, "sens_batch": 256,
    "ramp_from": RAMP_FROM, "zero_by": BYPASS_FROM,
    "stepper": {"sigma_max": 1.0},          # their defaults, untouched
    "gap": {"cadence": 100, "window": 3, "eps": 0.0, "patience": 2,
            "blackout": 2, "min_dwell_updates": 391,
            "max_dwell_updates": DWELL * 391},
}

COMMON = dict(seed=SEED, resolution="R32", reduction="input_bilinear",
              mask="all19", epochs=EPOCHS, eval_every=4,
              eval_extra=[BYPASS_FROM], resolution_table=R32,
              checkpoint_epochs=[30, BYPASS_FROM, EPOCHS])

CELLS = []
for tag, lr in (("lo", 0.005), ("hi", 0.05)):
    CELLS += [
        {**COMMON, "id": f"plain_{tag}", "cell_id": f"plain_{tag}__seed0",
         "group": tag, "lr": lr, "gaussian": "Gnone", "operator": "none",
         "gaussian_table": None, "adaptive": False},
        {**COMMON, "id": f"gplateau_{tag}", "cell_id": f"gplateau_{tag}__seed0",
         "group": tag, "lr": lr, "gaussian": "Gplateau", "operator": "gaussian",
         "gaussian_table": GPLATEAU, "adaptive": False},
        {**COMMON, "id": f"adaptgap_{tag}", "cell_id": f"adaptgap_{tag}__seed0",
         "group": tag, "lr": lr, "gaussian": "Gplateau", "operator": "gaussian",
         "gaussian_table": GPLATEAU, "adaptive": True, "trigger": "gap",
         "adaptive_params": ADAPT},
    ]


def fix_data_layout(work: Path) -> str:
    """Expose a flat CIFAR dataset under a ``cifar-10-batches-py`` directory."""
    for probe in sorted(glob.glob("/kaggle/input/**/data_batch_1", recursive=True)):
        src = Path(probe).parent
        dest = work / "_data"
        dest.mkdir(parents=True, exist_ok=True)
        link = dest / "cifar-10-batches-py"
        if not link.exists():
            link.symlink_to(src, target_is_directory=True)
        log("data: %s -> %s" % (link, src))
        return str(dest)
    for hit in sorted(glob.glob("/kaggle/input/**/cifar-10-batches-py",
                                recursive=True)):
        return str(Path(hit).parent)          # already correctly shaped
    # Diagnose rather than fail blind: print what is actually mounted.
    for d in sorted(glob.glob("/kaggle/input/*")):
        log("mounted: %s" % d)
        for sub in sorted(glob.glob(d + "/*"))[:12]:
            log("    %s" % sub)
            for sub2 in sorted(glob.glob(sub + "/*"))[:6]:
                log("        %s" % sub2)
    raise SystemExit("CIFAR-10 not found in /kaggle/input")


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def build_assets(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    bundle = build_dataset(DataConfig(root=os.environ["STUDY_DATA"],
                                      download=False, num_val=0))
    labels = bundle.train.labels.numpy()
    rng = np.random.default_rng(0)
    picks = [rng.permutation(np.flatnonzero(labels == c))[:PER_CLASS]
             for c in range(bundle.num_classes)]
    subset = np.sort(np.concatenate(picks))
    assert subset.size == N_SUBSET
    g = numpy_generator(SEED, "batch")
    perms = np.stack([g.permutation(N_SUBSET).astype(np.int32)
                      for _ in range(EPOCHS)])
    train_probe = fixed_subset_indices(N_SUBSET, PROBE, 0, "study_train_probe",
                                       labels=labels[subset])
    np.savez_compressed(dest / "shared_indices.npz", subset=subset,
                        train_probe=train_probe, perm_seed0=perms)
    m = build_model(ModelConfig(arch="resnet20_bn_cifar"), bundle.num_classes,
                    seed=SEED)
    torch.save(m.state_dict(), dest / "init_seed0.pt")

    sd = torch.load(dest / "init_seed0.pt", map_location="cpu", weights_only=True)
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(np.ascontiguousarray(sd[k].numpy()).tobytes())
    got = {"subset": _sha(subset), "train_probe": _sha(train_probe),
           "perm_seed0_first30": _sha(perms[:30]), "init_seed0": h.hexdigest()}
    checks = {k: got[k] == v for k, v in EXPECTED.items()}
    z = np.load(dest / "shared_indices.npz")
    (dest / "assets_manifest.json").write_text(json.dumps(
        {"source_job": "job_long_convergence (self-generated, %d epochs)" % EPOCHS,
         "paired_with_campaign": all(checks.values()),
         "pairing_checks": checks, "expected": EXPECTED, "got": got,
         "note": ("perm_seed0 has %d rows, so its first 30 are checked against "
                  "the campaign digest: same generator stream, same minibatch "
                  "order over the first 30 epochs" % EPOCHS),
         "arrays": {k: {"sha256": _sha(z[k]), "shape": list(z[k].shape)}
                    for k in z.files},
         "states": {"init_seed0": h.hexdigest()}}, indent=2))
    for k, ok in checks.items():
        log("pairing %-20s %s" % (k, "MATCH" if ok else "DIFFER"))
    if not all(checks.values()):
        raise SystemExit("assets do not reproduce the campaign digests")
    log("assets OK: subset/probe/init exact, perms[:30] reproduce the campaign")
    return dest


if __name__ == "__main__":
    work = Path(os.environ.get("CAMPAIGN_WORK", "/kaggle/working"))
    os.environ["STUDY_DATA"] = fix_data_layout(work)
    assets = build_assets(work / "_assets_local")
    os.environ["CAMPAIGN_ASSETS"] = str(assets)
    log("Gplateau (scaled x%d): dwell %d, bypass %d, %d terminal epochs"
        % (SCALE, DWELL, BYPASS_FROM, EPOCHS - BYPASS_FROM))
    log("cells: %s" % [c["cell_id"] for c in CELLS])
    run_job(0, CELLS)

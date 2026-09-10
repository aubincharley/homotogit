"""Adaptive predictor-corrector continuation over per-site sigma -- first run.

One cell, seed 0, ResNet-20 + BatchNorm, official 50,000 / 10,000 CIFAR-10
split, 30 epochs, constant resolution 32:

* ``ADAPT__seed0`` -- ``AdaptiveSiteController``: train at fixed sigma until the
  gradient-norm EMA plateaus, measure ``dL/dsigma_l`` at all 19 sites on a fixed
  probe batch, take a sensitivity-scaled step toward zero, repeat.

No control cell is trained.  The shared assets regenerate the campaign's pinned
sha256 digests exactly (see PAIRING below), so the recorded campaign runs
``R32__Gplateau__input_bilinear__all19__seed0`` and ``R32__Gnone...__seed0`` are
already valid paired controls and retraining them would waste GPU budget.

Why resolution is held at 32
----------------------------
``AdaptiveSiteController.measure`` evaluates the sensitivity through
``pipe(images, 0.0)`` with no ``res=`` argument, so the probe runs at 32x32
whatever the training resolution is.  At constant ``r=32`` that is exactly the
training configuration and the measurement is consistent; under a resolution
schedule it would not be.  Holding resolution fixed also keeps exactly one
operator moving, so a difference is attributable to the adaptive rule.

Termination
-----------
Adaptivity is not allowed to miss the target objective.  From epoch
``ramp_from = 18`` the deadline overrides the adaptive rule with a linear ramp,
every site is exactly zero by ``zero_by = 21``, and the last nine epochs run the
ordinary unfiltered ResNet-20 -- the same terminal structure as the campaign's
fixed arms.  ``deadline_fired`` is recorded and a run that fires it must be
reported as partly a fixed schedule.

PAIRING -- verified, not assumed
--------------------------------
The campaign's pinned asset *dataset* is gone (gitignored locally, absent from
both Kaggle accounts).  ``stage_campaign_assets`` warns that initial weights
depend on the torch build and cannot be regenerated -- but they reproduce here:
regenerating subset / train_probe / perm_seed0 / init_seed0 from the recorded
construction yields the campaign's four pinned digests **exactly**.

:func:`assert_paired` re-checks all four inside the job and **aborts before
training** if any differs, so this arm is paired with the campaign by
verification rather than by seed number.  If it ever aborts, the assets are no
longer reproducible on that torch build and the run must not be compared to the
recorded numbers.

Paired controls, from the recorded campaign (EXP-011, seed 0):

    R32__Gnone__input_bilinear__all19__seed0      test acc 0.7513
    R32__Gplateau__input_bilinear__all19__seed0   test acc 0.7791

The adaptive arm shares their initial weights and per-epoch permutations, so the
seed-0 differences are paired differences.  One seed: no distribution, and the
campaign's measured single-seed noise floor is about +-0.1 pp.
"""
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
from scripts.campaign_driver import data_source, log, run_job

EPOCHS, SEED = 30, 0
PER_CLASS, N_SUBSET, PROBE = 5000, 50000, 500

ADAPTIVE_PARAMS = {
    "sigma_init": 1.0,          # same starting width as the plateau schedule
    "sens_batch": 256,          # fixed probe batch for dL/dsigma
    "ramp_from": 18,            # deadline ramp begins
    "zero_by": 21,              # every site exactly 0 -> 9 terminal epochs
    "tracker": {"beta": 0.9, "tol": 0.01, "patience": 30, "min_steps": 40},
    "stepper": {"delta_ref": 0.15, "dmin": 0.02, "dmax": 0.35, "sigma_max": 1.0},
}

CELLS = [
    {"id": "ADAPT", "cell_id": "ADAPT__seed0", "group": "ADAPT", "seed": SEED,
     "resolution": "R32", "gaussian": "Gplateau", "reduction": "input_bilinear",
     "mask": "all19", "operator": "gaussian",
     "adaptive": True, "adaptive_params": ADAPTIVE_PARAMS},
]

# sha256 of the completed campaign's shared assets.  Reproducing these is what
# makes this arm paired with the recorded runs; a matching seed number proves
# nothing.
EXPECTED_HASHES = {
    "subset": "33236cc6bd19fa6b89e06d441d3fcd8eb37dc8540f6a4f2b627b20af10894a41",
    "train_probe": "a33645e0e4f024d8c47c0c5bdd7fbc7ea59a976090a815c58ba029908f2c86bc",
    "perm_seed0": "5df32d0d922962e72fb4dd718a5eaad13f3209f75b8363aff64e57b89b4bb22f",
    "init_seed0": "6e652c49e5139e785887b9cb6b08ce5b0dee734bde6384550be6a8a3641e2c1b",
}


def _sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _sha_state(p):
    sd = torch.load(p, map_location="cpu", weights_only=True)
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].cpu().numpy()).tobytes())
    return h.hexdigest()


def build_assets(dest: Path) -> Path:
    """Generate this job's own shared assets and a self-describing manifest.

    Same construction as the campaign's: a stratified per-class subset from a
    fixed seed-0 permutation stream, one permutation per epoch from the ``batch``
    stream, a stratified fixed train probe, and seed-0 initial weights.  The
    manifest records that these are job-local, not the pinned campaign assets.
    """
    dest.mkdir(parents=True, exist_ok=True)
    bundle = build_dataset(DataConfig(root=data_source(), download=False, num_val=0))
    labels = bundle.train.labels.numpy()

    rng = np.random.default_rng(0)
    picks = [rng.permutation(np.flatnonzero(labels == c))[:PER_CLASS]
             for c in range(bundle.num_classes)]
    subset = np.sort(np.concatenate(picks))
    assert subset.size == N_SUBSET, subset.size
    counts = np.bincount(labels[subset], minlength=bundle.num_classes).tolist()
    assert counts == [PER_CLASS] * bundle.num_classes, counts

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

    z = np.load(dest / "shared_indices.npz")
    man = {
        "source_job": "job_adaptive_sigma (self-generated)",
        "source_study": "NOT the pinned campaign assets -- see module docstring",
        "paired_with_campaign": False,
        "torch": torch.__version__,
        "files": {n: {"sha256": hashlib.sha256((dest / n).read_bytes()).hexdigest(),
                      "bytes": (dest / n).stat().st_size}
                  for n in ("init_seed0.pt", "shared_indices.npz")},
        "arrays": {k: {"sha256": _sha_array(z[k]), "shape": list(z[k].shape),
                       "dtype": str(z[k].dtype)} for k in z.files},
        "states": {"init_seed0": _sha_state(dest / "init_seed0.pt")},
    }
    (dest / "assets_manifest.json").write_text(json.dumps(man, indent=2))
    log("generated job-local assets in %s" % dest)
    for k, v in man["arrays"].items():
        log("  %-12s %s %s" % (k, v["sha256"][:16], v["shape"]))
    log("  init_seed0   %s" % man["states"]["init_seed0"][:16])
    return dest


def assert_paired(assets: Path, out_dir: Path) -> dict:
    """Abort before training unless all four campaign digests reproduce."""
    man = json.loads((assets / "assets_manifest.json").read_text())
    got = {k: v["sha256"] for k, v in man["arrays"].items()}
    got["init_seed0"] = man["states"]["init_seed0"]
    checks = {k: (got.get(k) == v) for k, v in EXPECTED_HASHES.items()}
    res = {"checks": checks, "all_match": all(checks.values()),
           "expected": EXPECTED_HASHES, "got": got}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairing_verification.json").write_text(json.dumps(res, indent=2))
    for k, ok in checks.items():
        log("pairing %-12s %s" % (k, "MATCH" if ok else "DIFFER"))
    if not res["all_match"]:
        raise SystemExit("assets do not reproduce the campaign digests; this arm "
                         "would not be paired -- refusing to train")
    log("paired with the recorded campaign: all four digests reproduce")
    return res


if __name__ == "__main__":
    work = Path(os.environ.get("CAMPAIGN_WORK", "/kaggle/working"))
    assets = build_assets(work / "_assets_local")
    assert_paired(assets, work / "_pairing")
    os.environ["CAMPAIGN_ASSETS"] = str(assets)
    log("cells: %s" % [c["cell_id"] for c in CELLS])
    run_job(0, CELLS)

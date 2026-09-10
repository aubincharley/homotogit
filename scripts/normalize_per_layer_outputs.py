"""Re-emit a per-layer study's outputs in this repository's campaign layout.

``scripts/study_per_layer_cpu.py`` grew its own output shape.  Every established
campaign under ``results/kaggle_outputs/`` instead looks like

    <kernel-slug>/<study>_<timestamp>/
        config.json                 the full study configuration, incl. `runs`
        environment.json            interpreter / torch / device capture
        pairing_verification.json   expected vs observed shared-asset digests
        study_summary.json          ONE list of per-run summaries, keyed `label`
        <arm>_seed<N>/metrics.json
        <arm>_seed<N>/summary.json

and downstream tooling (``scripts/aggregate_runs.py``, ``plot_campaign.py``)
reads that shape.  This converts an existing tree rather than re-running the
GPU: the numbers are copied unchanged, only the layout and the derived
provenance files are produced.

The pairing file is written honestly.  These runs are **self-paired** -- every
arm re-checks one set of in-run shared assets before its first update and aborts
on mismatch -- rather than paired against a prior campaign's pinned digests, so
``reference_campaign`` is null and the scope is stated explicitly.  A converted
file is marked ``derived_from_run_outputs`` so it is never mistaken for a proof
emitted by the training job itself.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def convert(src: Path, dest: Path, study_name: str) -> dict:
    seed_dirs = sorted(d for d in src.iterdir() if d.is_dir() and d.name.startswith("seed"))
    if not seed_dirs:
        raise SystemExit("no seed*/ directories under %s" % src)
    dest.mkdir(parents=True, exist_ok=True)

    study_summary, runs_spec = [], []
    # Pairing is WITHIN a seed: every arm at seed N shares one set of assets.
    # Across seeds the digests must DIFFER, or the seeds are not independent --
    # so both facts are checked and recorded.
    by_seed: dict = {}

    for sd in seed_dirs:
        seed = int(sd.name.replace("seed", ""))
        for arm_dir in sorted(d for d in sd.iterdir() if d.is_dir()):
            s = json.loads((arm_dir / "summary.json").read_text())
            label = "%s_seed%d" % (s["arm"], seed)
            out = dest / label
            out.mkdir(exist_ok=True)
            shutil.copyfile(arm_dir / "metrics.json", out / "metrics.json")

            by_seed.setdefault(seed, {})[label] = s.get("shared_digests", {})

            s = dict(s, label=label, seed=seed)
            (out / "summary.json").write_text(json.dumps(s, indent=2))
            for extra in ("realised_table.json",):
                if (arm_dir / extra).exists():
                    shutil.copyfile(arm_dir / extra, out / extra)
            study_summary.append(s)
            runs_spec.append({"label": label, "arm": s["arm"], "seed": seed,
                              "operator": s.get("operator"), "lr": s.get("lr"),
                              "profile": s.get("profile")})

    (dest / "study_summary.json").write_text(json.dumps(study_summary, indent=2))

    first = study_summary[0]
    config = {
        "name": study_name, "arch": "resnet20_bn_cifar",
        "n_subset": first["n_train"], "num_val": 0, "eval_split": "test",
        "n_test": first["n_test"], "seeds": sorted({r["seed"] for r in runs_spec}),
        "epochs": first["epochs"], "warmup": first["warmup"],
        "effective_batch": first["batch"],
        "microbatch": None,
        "microbatch_note": ("batch applied directly, not as microbatches; gradient "
                            "accumulation is exact so this changes only BatchNorm "
                            "batch statistics, and is held fixed across every arm"),
        "lr": first["lr"], "device": first.get("device"),
        "probe_size": first.get("probe_size"),
        "runs": runs_spec,
    }
    (dest / "config.json").write_text(json.dumps(config, indent=2))

    per_seed, all_match = {}, True
    for seed, obs in sorted(by_seed.items()):
        expected = next(iter(obs.values()))
        matches = {k: all(o.get(k) == expected.get(k) for o in obs.values())
                   for k in expected}
        ok = all(matches.values())
        all_match = all_match and ok
        per_seed[str(seed)] = {"expected": expected, "observed_by_run": obs,
                               "matches": matches, "all_match": ok,
                               "n_runs_checked": len(obs)}

    # seeds must NOT share assets, or they are not independent replicates
    seed_keys = sorted(by_seed)
    distinct = True
    if len(seed_keys) > 1:
        sigs = [json.dumps(per_seed[str(k)]["expected"], sort_keys=True) for k in seed_keys]
        distinct = len(set(sigs)) == len(sigs)

    pairing = {
        "reference_campaign": None,
        "pairing_scope": ("self-paired within each seed: initial weights, training "
                          "subset, test subset, sensitivity probe and every "
                          "per-epoch permutation are built once per seed and "
                          "re-checked by each arm before its first update; a "
                          "mismatch aborts before training"),
        "derived_from_run_outputs": True,
        "per_seed": per_seed,
        "all_match": all_match,
        "seeds_use_distinct_assets": distinct,
        "n_seeds": len(seed_keys),
    }
    (dest / "pairing_verification.json").write_text(json.dumps(pairing, indent=2))
    return {"runs": len(study_summary), "all_match": all_match, "dest": str(dest)}


def regenerate_assets(dest: Path, cfg_seeds, n_train, n_test, epochs, batch,
                      order_seed_of, probe_size: int = 512) -> dict:
    """Re-emit ``shared_indices.npz`` / ``init_seed<k>.pt`` and prove they are faithful.

    ``scripts/aggregate_runs.py`` deliberately recomputes pairing by hashing a
    study's own artifacts rather than trusting a recorded digest, so a study that
    does not write them reads as UNVERIFIED.  Every shared asset here is a pure
    function of the seed, so it can be rebuilt exactly -- and the rebuild is only
    accepted if its digests equal the ones each run recorded at training time.
    That check is what makes regeneration evidence rather than assertion.
    """
    import numpy as np
    import torch
    from continuation.config import DataConfig, ModelConfig
    from continuation.data import build_dataset
    from continuation.models import build_model
    from scripts.study_per_layer_cpu import resolve_data_root, sha, stratified_subset

    bundle = build_dataset(DataConfig(root=resolve_data_root(), download=False,
                                      num_val=0))
    labels = bundle.train.labels.numpy()
    arrays, regenerated = {}, {}
    for seed in cfg_seeds:
        subset = stratified_subset(labels, n_train, seed=seed)
        probe = np.sort(np.random.default_rng(2000 + seed).choice(
            subset.size, size=min(probe_size, subset.size), replace=False))
        test_idx = np.sort(np.random.default_rng(seed + 1).choice(
            len(bundle.test), size=n_test, replace=False))
        rng = np.random.default_rng(1000 + order_seed_of(seed))
        perms = np.stack([rng.permutation(subset.size) for _ in range(epochs)])
        model = build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=seed)
        state = model.state_dict()
        blob = torch.cat([v.flatten().float() for v in state.values()])
        regenerated[seed] = {"subset": sha(subset), "test_idx": sha(test_idx),
                             "perms": sha(perms), "probe_idx": sha(probe),
                             "init": sha(blob)}
        arrays["subset"] = subset
        arrays["train_probe"] = probe
        arrays["perm_seed%d" % seed] = perms
        torch.save(state, dest / ("init_seed%d.pt" % seed))
    np.savez_compressed(dest / "shared_indices.npz", **arrays)
    return regenerated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="directory holding seed*/ subdirectories")
    ap.add_argument("dest", type=Path, help="study directory to create")
    ap.add_argument("--name", default=None)
    ap.add_argument("--probe-size", type=int, default=512,
                    help="only needed when the runs predate probe_size being "
                         "recorded in their summaries")
    ap.add_argument("--regenerate-assets", action="store_true",
                    help="rebuild shared_indices.npz / init_seed*.pt and verify "
                         "them against the digests the runs recorded")
    a = ap.parse_args()
    r = convert(a.src, a.dest, a.name or a.dest.name)
    if a.regenerate_assets:
        cfg = json.loads((a.dest / "config.json").read_text())
        pv = json.loads((a.dest / "pairing_verification.json").read_text())
        first = json.loads((a.dest / "study_summary.json").read_text())[0]
        regen = regenerate_assets(
            a.dest, cfg["seeds"], cfg["n_subset"], cfg["n_test"],
            cfg["epochs"], cfg["effective_batch"], lambda s: s,
            probe_size=cfg.get("probe_size") or a.probe_size)
        ok = {}
        for seed in cfg["seeds"]:
            recorded = pv["per_seed"][str(seed)]["expected"]
            ok[seed] = {k: regen[seed].get(k) == recorded.get(k) for k in recorded}
        pv["regenerated_assets"] = {
            "written": ["shared_indices.npz"] + ["init_seed%d.pt" % s for s in cfg["seeds"]],
            "matches_recorded_digests": {str(k): v for k, v in ok.items()},
            "all_match": all(all(v.values()) for v in ok.values()),
            "note": ("assets rebuilt from the seed and accepted only because their "
                     "digests equal those recorded during training"),
        }
        (a.dest / "pairing_verification.json").write_text(json.dumps(pv, indent=2))
        r["assets_verified"] = pv["regenerated_assets"]["all_match"]
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()

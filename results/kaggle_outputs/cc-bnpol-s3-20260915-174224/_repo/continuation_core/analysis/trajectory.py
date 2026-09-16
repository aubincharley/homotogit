"""Export optimisation trajectories from saved checkpoints.

``export_trajectories([run_a, run_b], out)`` writes one ``.npz`` holding, for
each run, the parameter vectors of all its checkpoints (float32, trainable
parameters in ``named_parameters`` order), their global updates and epochs,
and a JSON side-car with the intervention state used before and after each
checkpoint.  Runs are **paired** when they share seed and pinned initial state;
the export records that check rather than assuming it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..checkpoint import list_checkpoints, load_checkpoint
from ..config import ExperimentConfig
from ..models import build_model
from .params import parameter_names, to_vector


def export_trajectories(run_dirs, out_path, num_classes: int = 10, in_channels: int = 3) -> dict:
    out_path = Path(out_path)
    arrays, meta = {}, {"runs": []}
    init_digests = set()
    names_ref = None
    for k, run in enumerate(run_dirs):
        ckpts = list_checkpoints(run)
        if not ckpts:
            raise FileNotFoundError("no checkpoints under %s/checkpoints" % run)
        first = load_checkpoint(ckpts[0])
        cfg = ExperimentConfig.from_dict(first["config"])
        model = build_model(cfg.model.arch, num_classes, in_channels, **cfg.model.options)
        names = parameter_names(model)
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            raise ValueError("runs have different parameter layouts; not comparable")
        vecs, rows = [], []
        for p in ckpts:
            ck = load_checkpoint(p)
            vecs.append(to_vector(ck["model_state"], names, dtype=torch.float32).numpy())
            rows.append({"file": p.name, "global_update": int(ck["global_update"]),
                         "epochs_completed": ck["epochs_completed"], "reason": ck["reason"],
                         "intervention": ck["intervention"]})
        prov = first.get("provenance", {})
        init_digests.add(json.dumps({"seed": cfg.run.seed,
                                     "states": prov.get("asset_states"),
                                     "arrays": prov.get("asset_arrays")}, sort_keys=True))
        arrays["run%d_vectors" % k] = np.stack(vecs)
        arrays["run%d_updates" % k] = np.asarray([r["global_update"] for r in rows])
        meta["runs"].append({"index": k, "dir": str(run), "method": cfg.method_spec().id,
                             "seed": cfg.run.seed, "checkpoints": rows})
    meta["parameter_names"] = names_ref
    meta["paired"] = len(init_digests) == 1
    meta["pairing_note"] = ("identical seed and pinned asset digests" if meta["paired"]
                            else "runs differ in seed or asset digests: not paired")
    np.savez(out_path, **arrays)
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def load_trajectories(path) -> tuple:
    z = np.load(path)
    meta = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    return z, meta

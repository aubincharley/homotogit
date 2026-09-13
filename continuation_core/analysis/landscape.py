"""Loss at fixed weights under chosen intervention states and BN policies.

Built for three questions, without assuming the answer:

1. At identical weights, how does changing the intervention change the loss?
   -> ``CheckpointEvaluator.loss`` with ``state="used"``, ``"next"``,
   ``"target"`` or any explicit ``InterventionState``.
2. How do optimisation trajectories differ, especially at transitions?
   -> ``continuation_core.analysis.trajectory`` + ``pca_plane`` (common plane,
   explained variance, per-point projection residuals).
3. How sensitive are final solutions to weight perturbations?
   -> ``perturbation_sensitivity`` with filter-normalised directions.

The checkpoint file is read once and never written; the evaluator builds its
own model, loads a *copy* of the weights, and evaluates through
``continuation_core.evaluate.evaluate``, which restores model mode, controller
state and RNG afterwards.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..checkpoint import load_checkpoint
from ..config import ExperimentConfig
from ..controller import InterventionController, InterventionState
from ..data import build_pipeline, load_dataset
from ..evaluate import evaluate
from ..models import build_model, site_map
from .params import filter_normalized_direction, from_vector, parameter_names, to_vector


class CheckpointEvaluator:
    def __init__(self, checkpoint_path, *, dataset=None, device="cpu",
                 split: str = "train_probe", n_images: int | None = None,
                 batch_size: int = 500, assets=None):
        from .. import assets as assets_mod
        self.path = str(checkpoint_path)
        self.ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
        self.cfg = ExperimentConfig.from_dict(self.ckpt["config"])
        self.device = torch.device(device)
        self.dataset = dataset or load_dataset(self.cfg.data)
        self.pipeline = build_pipeline(self.dataset, self.cfg.data).to(self.device)
        arch = self.cfg.model.arch
        self.model = build_model(arch, self.dataset.num_classes,
                                 int(self.dataset.train.images.shape[1]),
                                 **self.cfg.model.options).to(self.device)
        self.model.load_state_dict({k: v.clone() for k, v in self.ckpt["model_state"].items()})
        self.model.eval()
        self.controller = InterventionController(self.cfg.method_spec(), site_map(arch), arch)
        self.controller.attach(self.model)
        self.names = parameter_names(self.model)
        self.batch_size = int(batch_size)
        self.split = split
        if split == "test":
            x, y = self.dataset.test.images, self.dataset.test.labels
        else:
            a = assets or assets_mod.load(self.cfg.assets.dir, self.cfg.run.seed,
                                          verify_first=False)
            sub = torch.as_tensor(a["subset"], dtype=torch.long)
            x, y = self.dataset.train.images[sub], self.dataset.train.labels[sub]
            if split == "train_probe":
                p = torch.as_tensor(a["train_probe"], dtype=torch.long)
                x, y = x[p], y[p]
            elif split != "train":
                raise ValueError("split must be train_probe, train or test")
        if n_images is not None:
            x, y = x[:n_images], y[:n_images]
        self.images, self.labels = x.to(self.device), y.to(self.device)

    # -- states ------------------------------------------------------------

    def state(self, which) -> InterventionState:
        if isinstance(which, InterventionState):
            return which
        iv = self.ckpt["intervention"]
        if which in ("used", "current"):
            rec = iv["used_for_last_update"]
        elif which == "next":
            rec = iv["next_update"]
        elif which == "target":
            return self.controller.target_state()
        else:
            raise ValueError("state must be used/current, next, target or an InterventionState")
        if rec is None:
            raise ValueError("checkpoint has no %r state (update 0 or end of budget)" % which)
        return InterventionState.from_dict({**rec["state"], "label": which})

    def base_params(self) -> dict:
        return {n: p.detach().clone() for n, p in self.model.named_parameters()}

    def base_vector(self) -> torch.Tensor:
        return to_vector(dict(self.model.named_parameters()), self.names)

    # -- loss --------------------------------------------------------------

    def loss(self, *, params: dict | None = None, vector: torch.Tensor | None = None,
             state="target", bn_policy: str = "running_stats") -> dict:
        if vector is not None:
            params = from_vector(vector, dict(self.model.named_parameters()), self.names,
                                 device=self.device)
        st = self.state(state)
        r = evaluate(self.model, self.controller, self.pipeline, self.images, self.labels,
                     st, bn_policy=bn_policy, batch_size=self.batch_size,
                     params=params, split=self.split)
        r["checkpoint"] = self.path
        r["global_update"] = int(self.ckpt["global_update"])
        r["weights"] = "checkpoint" if params is None else "override"
        return r


# --------------------------------------------------------------------------

def pca_plane(vectors, center="mean") -> dict:
    """Common 2-D plane through a set of parameter vectors (rows).

    ``center``: ``"mean"``, ``"last"`` or an integer row index used as origin.
    Returns the origin, two orthonormal directions, the explained-variance ratio
    of every principal component, the in-plane coordinates of every point and
    its projection residual (absolute and relative to its distance from the
    origin).  A plane that explains little variance, or points with large
    residuals, mean the 2-D picture is not representative of the trajectory.
    """
    X = torch.as_tensor(np.asarray(vectors), dtype=torch.float64)
    if X.dim() != 2 or X.shape[0] < 3:
        raise ValueError("need at least 3 vectors of equal length")
    if center == "mean":
        origin = X.mean(0)
    elif center == "last":
        origin = X[-1]
    else:
        origin = X[int(center)]
    C = X - origin
    _, S, Vt = torch.linalg.svd(C - C.mean(0) if center == "mean" else C,
                                full_matrices=False)
    var = S ** 2
    d1, d2 = Vt[0], Vt[1]
    coords = torch.stack([C @ d1, C @ d2], dim=1)
    recon = coords[:, :1] * d1 + coords[:, 1:] * d2
    resid = (C - recon).norm(dim=1)
    dist = C.norm(dim=1)
    return {"origin": origin, "d1": d1, "d2": d2,
            "explained_variance_ratio": (var / var.sum()).tolist(),
            "plane_variance_ratio": float((var[:2].sum() / var.sum())),
            "coords": coords.tolist(), "residual_norm": resid.tolist(),
            "relative_residual": (resid / dist.clamp_min(1e-30)).tolist(),
            "center": center if isinstance(center, str) else int(center)}


def plane_grid(evaluator: CheckpointEvaluator, origin, d1, d2, alphas, betas,
               states=("target",), bn_policies=("fixed_batch_stats",)) -> dict:
    """Loss on ``origin + a*d1 + b*d2`` for every (state, BN policy)."""
    origin = torch.as_tensor(origin, dtype=torch.float64)
    d1 = torch.as_tensor(d1, dtype=torch.float64)
    d2 = torch.as_tensor(d2, dtype=torch.float64)
    out = {"alphas": list(map(float, alphas)), "betas": list(map(float, betas)),
           "surfaces": []}
    for st in states:
        for pol in bn_policies:
            ce = np.full((len(alphas), len(betas)), np.nan)
            acc = np.full_like(ce, np.nan)
            label = None
            for i, a in enumerate(alphas):
                for j, b in enumerate(betas):
                    r = evaluator.loss(vector=origin + float(a) * d1 + float(b) * d2,
                                       state=st, bn_policy=pol)
                    ce[i, j], acc[i, j], label = r["ce"], r["acc"], r["state"]
            out["surfaces"].append({"state": label, "bn_policy": pol,
                                    "ce": ce.tolist(), "acc": acc.tolist()})
    return out


def perturbation_sensitivity(evaluator: CheckpointEvaluator, epsilons, n_directions: int,
                             seed: int = 0, states=("target",),
                             bn_policies=("running_stats",)) -> dict:
    """Loss change under filter-normalised random perturbations of the weights.

    ``w + eps * d`` with ``d`` from :func:`filter_normalized_direction`, so
    ``eps`` is a fraction of each filter's own norm.  Directions are shared
    across states and BN policies (same generator seed), so differences between
    states are paired.
    """
    g = torch.Generator().manual_seed(int(seed))
    base = {n: p.detach().cpu() for n, p in evaluator.model.named_parameters()}
    dirs = [filter_normalized_direction(base, evaluator.names, g) for _ in range(n_directions)]
    results = []
    for st in states:
        for pol in bn_policies:
            ref = evaluator.loss(state=st, bn_policy=pol)
            rows = []
            for eps in epsilons:
                deltas = []
                for d in dirs:
                    params = {n: (base[n].to(torch.float64) + float(eps) * d[n])
                              .to(base[n].dtype).to(evaluator.device) for n in evaluator.names}
                    deltas.append(evaluator.loss(params=params, state=st,
                                                 bn_policy=pol)["ce"] - ref["ce"])
                rows.append({"epsilon": float(eps), "delta_ce_mean": float(np.mean(deltas)),
                             "delta_ce_min": float(np.min(deltas)),
                             "delta_ce_max": float(np.max(deltas)),
                             "delta_ce_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
                             "deltas": deltas})
            results.append({"state": ref["state"], "bn_policy": pol, "base_ce": ref["ce"],
                            "base_acc": ref["acc"], "rows": rows})
    return {"checkpoint": evaluator.path, "normalization": "filter-wise (Li et al. 2018), "
            "1-D tensors not perturbed", "n_directions": n_directions, "seed": seed,
            "split": evaluator.split, "n_images": int(evaluator.images.shape[0]),
            "results": results}

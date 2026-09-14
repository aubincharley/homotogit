"""Command line.  ``py -m continuation_core <command> --help`` for options.

describe           method definition, schedule table, per-site sigma, sites
dry-run            build everything, verify assets, forward/backward at every
                   distinct state, check the target-state bypass; no training
train              train a preset (or a JSON config); resumable
evaluate           loss/accuracy of a checkpoint under a chosen state and BN policy
export-trajectory  parameter vectors of all checkpoints of one or more runs
pca-plane          common PCA plane of exported trajectories
plane-loss         loss on a 2-D plane under several states / BN policies
perturb            loss change under filter-normalised weight perturbations
verify-assets      recompute pinned-asset digests
make-assets        generate a NEW asset set for another dataset / model
                   (--init-from reuses an existing set's initial weights)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _parse_state(text):
    from .controller import InterventionState
    if text in ("used", "current", "next", "target"):
        return text
    kv = dict(item.split("=", 1) for item in text.split(","))
    r = kv.get("r", kv.get("resolution"))
    s = kv.get("sigma", kv.get("G"))
    return InterventionState(None if r in (None, "none") else int(r),
                             None if s in (None, "none") else float(s), label=text)


def _config(args):
    from .config import ExperimentConfig
    from .presets import reference
    if getattr(args, "config", None):
        cfg = ExperimentConfig.load(args.config)
    else:
        cfg = reference(args.method, args.seed, data_root=args.data_root,
                        assets_dir=args.assets, out_dir=args.out, device=args.device)
    if getattr(args, "epochs", None) is not None and args.epochs != cfg.budget.epochs:
        cfg.budget.epochs = args.epochs
        cfg.validation_status = "modified"
        cfg.notes.append("budget changed on the command line")
    if getattr(args, "transition_offsets", None):
        cfg.checkpoint.transition_offsets = tuple(int(v) for v in args.transition_offsets.split(","))
    if getattr(args, "every_updates", None):
        cfg.checkpoint.every_updates = int(args.every_updates)
    return cfg


def _dump(obj, out):
    text = json.dumps(obj, indent=2, allow_nan=True)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n", encoding="utf-8")
        print("wrote", out)
    else:
        print(text)


def cmd_describe(args):
    from .controller import InterventionController
    from .methods import get_method
    from .models import site_map
    m = get_method(args.method)
    sm = site_map(args.arch)
    ctrl = InterventionController(m, sm, args.arch)
    rows = []
    for e in range(args.epochs):
        ctrl.set_epoch(e)
        rows.append({"epoch": e, "resolution": ctrl.state.resolution,
                     "G": ctrl.state.sigma, "scale": ctrl.scale(),
                     "per_site_sigma": sorted(set(ctrl.per_site_sigma()))})
    ctrl.set_state(ctrl.target_state())
    placement = m.gaussian.placement if m.gaussian else None
    _dump({"method": m.to_dict(), "arch": args.arch,
           "gaussian_sites": sm[placement] if placement else [],
           "reduction_point": sm["reduction"][m.resolution.point] if m.resolution else None,
           "schedule": rows, "target_state": ctrl.describe_state()}, args.out)


def cmd_dry_run(args):
    import torch.nn.functional as F
    from .models import build_model
    from .train import Trainer
    cfg = _config(args)
    tr = Trainer(cfg, device=args.device)
    x = tr.pipeline(tr.train_images[:32])
    y = tr.train_labels[:32]
    seen, report = set(), {"states": []}
    for e in range(cfg.budget.epochs):
        st = tr.controller.state_for_epoch(e)
        if (st.resolution, st.sigma) in seen:
            continue
        seen.add((st.resolution, st.sigma))
        tr.controller.set_state(st)
        tr.model.zero_grad(set_to_none=True)
        logits = tr.model(x)
        F.cross_entropy(logits, y).backward()
        g = all(torch.isfinite(p.grad).all() for p in tr.model.parameters() if p.grad is not None)
        report["states"].append({"first_epoch": e, "resolution": st.resolution, "G": st.sigma,
                                 "per_site_sigma": sorted(set(tr.controller.per_site_sigma())),
                                 "logits_finite": bool(torch.isfinite(logits).all()),
                                 "grads_finite": bool(g)})
    tr.model.zero_grad(set_to_none=True)
    tr.controller.set_state(tr.controller.target_state())
    plain = build_model(cfg.model.arch, tr.dataset.num_classes, int(x.shape[1]),
                        **cfg.model.options).to(tr.device)
    plain.load_state_dict(tr.model.state_dict())
    tr.model.eval(), plain.eval()
    with torch.no_grad():
        diff = float((tr.model(x) - plain(x)).abs().max())
    tr.model.train()
    report.update(transition_updates=tr.transition_updates,
                  extra_checkpoints=tr.extra_checkpoints,
                  updates_per_epoch=tr.updates_per_epoch, total_updates=tr.total_updates,
                  target_bypass_max_abs_diff=diff, target_bypass_bitwise=diff == 0.0,
                  assets_verified=bool(tr.assets.get("verification")),
                  validation_status=cfg.validation_status)
    _dump(report, args.report)
    return 0 if report["target_bypass_bitwise"] else 1


def cmd_train(args):
    from .train import Trainer
    cfg = _config(args)
    tr = Trainer(cfg, device=args.device)
    if args.resume:
        tr.resume(args.resume)
    out = tr.run(max_updates=args.max_updates)
    if out is None:
        path = tr.save("rolling", name="rolling.pt")
        print("stopped after update %d; resume with --resume %s" % (tr.global_update, path))
    else:
        print(json.dumps(out["final"], indent=2))


def cmd_evaluate(args):
    from .analysis import CheckpointEvaluator
    ev = CheckpointEvaluator(args.checkpoint, device=args.device, split=args.split,
                             n_images=args.n_images, batch_size=args.batch_size)
    rows = [ev.loss(state=_parse_state(s), bn_policy=p)
            for s in args.state.split(";") for p in args.bn_policy.split(",")]
    _dump(rows, args.out)


def cmd_export(args):
    from .analysis import export_trajectories
    _dump(export_trajectories(args.runs, args.out), None)


def cmd_pca(args):
    import numpy as np
    from .analysis import load_trajectories, pca_plane
    z, meta = load_trajectories(args.trajectory)
    idx = [int(i) for i in args.runs.split(",")] if args.runs else range(len(meta["runs"]))
    X = np.concatenate([z["run%d_vectors" % i] for i in idx]).astype(np.float64)
    center = args.center if args.center in ("mean", "last") else int(args.center)
    plane = pca_plane(X, center=center)
    torch.save({k: plane[k] for k in ("origin", "d1", "d2")}, args.out)
    stats = {k: v for k, v in plane.items() if k not in ("origin", "d1", "d2")}
    stats["rows"] = [{"run": i, "global_update": int(u)} for i in idx
                     for u in z["run%d_updates" % i]]
    _dump(stats, Path(args.out).with_suffix(".json"))


def cmd_plane_loss(args):
    import numpy as np
    from .analysis import CheckpointEvaluator, filter_normalized_direction, plane_grid, to_vector
    ev = CheckpointEvaluator(args.checkpoint, device=args.device, split=args.split,
                             n_images=args.n_images, batch_size=args.batch_size)
    if args.plane:
        p = torch.load(args.plane, map_location="cpu", weights_only=False)
        origin, d1, d2, kind = p["origin"], p["d1"], p["d2"], "pca"
    else:
        g = torch.Generator().manual_seed(args.random_seed)
        base = {n: t.detach().cpu() for n, t in ev.model.named_parameters()}
        d1 = to_vector(filter_normalized_direction(base, ev.names, g), ev.names)
        d2 = to_vector(filter_normalized_direction(base, ev.names, g), ev.names)
        origin, kind = ev.base_vector(), "filter-normalised random, centred on checkpoint"
    lo, hi, n = args.grid.split(":")
    grid = np.linspace(float(lo), float(hi), int(n))
    out = plane_grid(ev, origin, d1, d2, grid, grid,
                     states=[_parse_state(s) for s in args.states.split(";")],
                     bn_policies=args.bn_policies.split(","))
    out.update(checkpoint=args.checkpoint, plane=kind, split=args.split,
               n_images=int(ev.images.shape[0]))
    _dump(out, args.out)


def cmd_perturb(args):
    from .analysis import CheckpointEvaluator, perturbation_sensitivity
    ev = CheckpointEvaluator(args.checkpoint, device=args.device, split=args.split,
                             n_images=args.n_images, batch_size=args.batch_size)
    _dump(perturbation_sensitivity(ev, [float(e) for e in args.epsilons.split(",")],
                                   args.directions, args.seed_directions,
                                   states=[_parse_state(s) for s in args.states.split(";")],
                                   bn_policies=args.bn_policies.split(",")), args.out)


def cmd_verify_assets(args):
    from .assets import verify
    _dump(verify(args.assets), args.out)


def cmd_make_assets(args):
    from .assets import make_assets
    from .config import DataConfig
    from .data import load_dataset
    from .models import build_model
    ds = load_dataset(DataConfig(name=args.dataset, root=args.data_root))
    man = make_assets(args.out, lambda: build_model(args.arch, ds.num_classes,
                                                    int(ds.train.images.shape[1])),
                      n_train=len(ds.train), epochs=args.epochs,
                      seeds=tuple(int(s) for s in args.seeds.split(",")),
                      probe_size=args.probe_size, init_from=args.init_from,
                      subset_size=args.subset_size,
                      indices_from=args.indices_from,
                      provenance={"dataset": args.dataset, "arch": args.arch})
    _dump({"written": args.out, "states": man["states"]}, None)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="continuation_core", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def run_opts(p):
        p.add_argument("--method", default="plain")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--data-root", default="data")
        p.add_argument("--assets", default="assets/cifar10_resnet20bn")
        p.add_argument("--out", default="runs")
        p.add_argument("--config", help="JSON ExperimentConfig instead of the reference preset")
        p.add_argument("--device", default="auto")
        p.add_argument("--epochs", type=int)
        p.add_argument("--transition-offsets", help="e.g. -50,-1,1,50")
        p.add_argument("--every-updates", type=int)

    def eval_opts(p):
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--split", default="train_probe", choices=("train_probe", "train", "test"))
        p.add_argument("--n-images", type=int)
        p.add_argument("--batch-size", type=int, default=500)
        p.add_argument("--out")

    p = sub.add_parser("describe")
    p.add_argument("--method", required=True)
    p.add_argument("--arch", default="resnet20_bn_cifar")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_describe)

    p = sub.add_parser("dry-run")
    run_opts(p)
    p.add_argument("--report", help="write the dry-run report to this JSON file")
    p.set_defaults(fn=cmd_dry_run)

    p = sub.add_parser("train")
    run_opts(p)
    p.add_argument("--max-updates", type=int)
    p.add_argument("--resume")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("evaluate")
    eval_opts(p)
    p.add_argument("--state", default="used;target",
                   help="';'-separated: used, next, target, or r=16,sigma=0.5")
    p.add_argument("--bn-policy", default="running_stats")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("export-trajectory")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("pca-plane")
    p.add_argument("--trajectory", required=True)
    p.add_argument("--runs", help="comma-separated run indices (default: all)")
    p.add_argument("--center", default="mean")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_pca)

    p = sub.add_parser("plane-loss")
    eval_opts(p)
    p.add_argument("--plane", help="file written by pca-plane; default: random filter-normalised")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--grid", default="-1:1:11")
    p.add_argument("--states", default="target")
    p.add_argument("--bn-policies", default="fixed_batch_stats")
    p.set_defaults(fn=cmd_plane_loss)

    p = sub.add_parser("perturb")
    eval_opts(p)
    p.add_argument("--epsilons", default="0.01,0.05,0.1")
    p.add_argument("--directions", type=int, default=5)
    p.add_argument("--seed-directions", type=int, default=0)
    p.add_argument("--states", default="target")
    p.add_argument("--bn-policies", default="running_stats")
    p.set_defaults(fn=cmd_perturb)

    p = sub.add_parser("verify-assets")
    p.add_argument("--assets", default="assets/cifar10_resnet20bn")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_verify_assets)

    p = sub.add_parser("make-assets")
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--arch", required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--probe-size", type=int, default=500,
                   help="train-probe size; 500 is 1%% of CIFAR-10 but 10%% of STL-10")
    p.add_argument("--subset-size", type=int,
                   help="train on this many images instead of the whole split, to "
                        "match another dataset's updates per epoch")
    p.add_argument("--indices-from",
                   help="asset directory to copy shared_indices.npz from, so a "
                        "different model sees the same images in the same order")
    p.add_argument("--init-from",
                   help="asset directory to copy init_seed<k>.pt from instead of "
                        "drawing new weights (checked with a strict load)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_make_assets)

    args = ap.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())

"""Every signature at every epoch, for the twelve reference cells.

``TRAJ_SHARD`` picks the slice.

The endpoint probes say what a curriculum *leaves*; they cannot say **when** the
separation appears or in which order the signatures emerge, and that ordering is
the difference between a correlation and a mechanism.  If sensitivity falls first
and curvature follows, the chain

    intervention -> sensitivity down -> curvature down -> gap down

is a candidate; if they move together, it is not.

The per-epoch checkpoints are written by the trainer already
(``CheckpointConfig.every_epoch`` is on in the reference preset), so this costs
no training.  Settings are lighter than the endpoint probes -- a hundred and
twenty points instead of twenty-eight -- and the analysis reads shapes over time,
not fine differences between neighbouring points.

It also carries the one prediction the endpoint measurements leave open: if the
resolution arm's frequency profile starts low and **returns** to the control's
while its sensitivity starts low and **stays** low, then the dissociation between
the two interventions is a transient-versus-persistent effect rather than an
anomaly.
"""
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def find(pattern, what):
    hits = sorted(glob.glob("/kaggle/input/**/" + pattern, recursive=True))
    if not hits:
        raise SystemExit("%s not found; attach the dataset" % what)
    return str(Path(hits[0]).parent)


def checkpoints(runs_dir):
    """``(cell, epoch, path)`` for the reference cells only."""
    pat = re.compile(r"epoch_(\d+)\.pt$")
    out = []
    for d in sorted(Path(runs_dir).iterdir()):
        if not d.is_dir() or "__reference__" not in d.name:
            continue
        for f in sorted((d / "checkpoints").glob("epoch_*.pt")):
            m = pat.search(f.name)
            if m:
                out.append((d.name, int(m.group(1)), str(f)))
    return sorted(out, key=lambda r: (r[0], r[1]))


def main(runs_dir, out_dir, shard, device="cuda"):
    from continuation_core.analysis import CheckpointEvaluator, curvature, sensitivity

    si, sn = (int(v) for v in shard.split("/"))
    todo = checkpoints(runs_dir)[si::sn]
    print("shard %s: %d points" % (shard, len(todo)), flush=True)
    t0, rows = time.perf_counter(), []
    for cell, epoch, path in todo:
        ev = CheckpointEvaluator(path, device=device, split="train", batch_size=1000)
        te = CheckpointEvaluator(path, device=device, split="test", batch_size=1000)
        r_tr, r_te = ev.loss(state="target"), te.loss(state="target")
        del te
        s = sensitivity.probe(ev, n_images=500, batch_size=250,
                              epsilons=(0.5, 1.0, 3.0))
        c = curvature.probe(ev, n_images=2000, batch_size=500, draws=24,
                            top_k=1, power_iters=25, seed=int(ev.cfg.run.seed))
        rows.append({"cell": cell, "epoch": epoch,
                     "train_err": 1.0 - r_tr["acc"], "test_err": 1.0 - r_te["acc"],
                     "gap": (1.0 - r_te["acc"]) - (1.0 - r_tr["acc"]),
                     "jac_frobenius": s["jacobian"]["frobenius_mean"],
                     "jac_spectral": s["jacobian"]["spectral_mean"],
                     "mean_radius": s["frequency"]["mean_radius"],
                     "frac_above_k8": s["frequency"]["fraction_above"]["8"],
                     "S@3": s["amplitude_curve"][-1]["displacement"],
                     "trace_H": c["trace"]["mean"],
                     "lambda_max": c["top_eigenvalues"][0]["eigenvalue"]})
        r = rows[-1]
        print("  %-38s ep%02d gap=%.4f |J|=%7.2f k=%5.2f trH=%8.1f  %5.0fs"
              % (cell, epoch, r["gap"], r["jac_frobenius"], r["mean_radius"],
                 r["trace_H"], time.perf_counter() - t0), flush=True)
        del ev
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / ("traj_shard%d.json" % si)).write_text(
        json.dumps({"shard": shard, "points": rows}, indent=2))
    print("done in %.0f s" % (time.perf_counter() - t0))


if __name__ == "__main__":
    shard = os.environ.get("TRAJ_SHARD", "0/4")
    find("cifar-10-batches-py", "CIFAR-10")
    runs = find("grid_plan.json", "trained grid runs")
    out = Path(os.environ.get("STUDY_OUT", "/kaggle/working"))
    main(str(Path(runs) / "runs"), str(out), shard)

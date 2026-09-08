#!/usr/bin/env python3
"""Draw the loss landscape of a finished run. Takes the folder the run lives in.

    python explore.py runs/latest                    # every figure it can make
    python explore.py runs/latest --only surface,ratio
    python explore.py runs/latest --against runs/20260907-1200_baseline60_resnet18
    python explore.py runs/latest --plot-only        # redraw, compute nothing

Needs checkpoints, which means the run must have been trained with ckpt_every > 0.
Measurements are appended to <run>/landscape.jsonl as they are computed, so a
sweep that dies keeps what it had and --plot-only can redraw from it later.

Deliberately NOT part of src/, for the same reason analyze.py is not: the kernel's
job is to produce numbers, and matplotlib should never have to exist inside it.
The computation this file drives does live in src/ -- cifarbase.landscape -- so a
Kaggle run can produce the measurements even though it can never draw them.

What each figure is for:

  surface.png     Two filter-normalised random directions around the final
                  weights. The standard picture (Li et al. 2018), and the one to
                  compare across arms: is the homotopy's minimum flatter?
  plane.png       The plane through three checkpoints of the run, with the whole
                  trajectory projected into it. Says whether following theta*(s)
                  stayed in one basin or hopped between them.
  s_alpha.png     Loss over (s, alpha): the homotopy parameter against a
                  displacement in weight space. The one figure here that is not
                  borrowed from somewhere else -- it shows the landscape
                  deforming as the residual branches switch on, which a
                  conventional surface plot has no axis for.
  barrier.png     Loss along the segment between two runs' solutions, BatchNorm
                  recomputed at every point. Linearly connected or not.
  ratio.png       rms(s*F) / rms(shortcut) per block over training. The test for
                  whether gamma_bn2 quietly absorbed s and the homotopy was
                  reparametrised away.
  curvature.png   Top Hessian eigenvalues and trace along the trajectory.
  loss_vs_s.png   L_s(theta_t) at fixed theta, swept over s.

Every loss here is measured on a fixed subset (--max-images), not the whole
split: a surface is hundreds of evaluations. The figures say so in their titles,
because a loss surface drawn on 5k images is not quite the objective that was
minimised and the plot should not pretend otherwise.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from analyze import CYCLE, MUTED, TEST, TRAIN, _theme

FIGURES = ("surface", "plane", "s_alpha", "loss_vs_s", "ratio", "curvature",
           "barrier")

# Loss spans orders of magnitude across a surface, so contours are log-spaced:
# linear levels put every line inside the basin and leave the walls blank.
CMAP = "viridis"


# --------------------------------------------------------------------------
# loading a run
# --------------------------------------------------------------------------

def load_invocation(directory):
    path = os.path.join(directory, "invocation.json")
    if not os.path.isfile(path):
        raise SystemExit(f"!! no invocation.json in {directory}: point this at a "
                         f"run directory made by main.py")
    with open(path) as fh:
        return json.load(fh)


def prepare(directory, split_name, seed=None):
    """Model, data split and checkpoints for a run. Imports torch lazily."""
    from cifarbase.data import load_cifar10
    from cifarbase.homotopy import ResidualGate
    from cifarbase.landscape import list_checkpoints, load_checkpoint
    from cifarbase.model import build_model
    from cifarbase.utils.device import pick_device

    cfg = load_invocation(directory)["config"]
    checkpoints = list_checkpoints(directory)
    if not checkpoints:
        raise SystemExit(
            f"!! no checkpoints in {directory}. Re-run with ckpt_every > 0 "
            f"(e.g. --ckpt-every 5); without saved weights there is no "
            f"trajectory to draw and no theta to sweep around.")

    seeds = sorted({c.get("seed", 0) for c in checkpoints})
    seed = seeds[0] if seed is None else seed
    if seed not in seeds:
        raise SystemExit(f"!! run has checkpoints for seeds {seeds}, not {seed}")
    checkpoints = [c for c in checkpoints if c.get("seed", 0) == seed]

    device = pick_device()
    model = build_model(cfg, device, quiet=True)
    gate = ResidualGate(model)
    train, _, test = load_cifar10(device, cfg)
    split = train if split_name == "train" else test

    # The last checkpoint is theta at the end of training: the point every
    # surface is centred on unless something else is asked for.
    load_checkpoint(checkpoints[-1]["path"], model, device)
    return {"cfg": cfg, "model": model, "gate": gate, "split": split,
            "train": train, "device": device, "checkpoints": checkpoints,
            "seed": seed, "dir": directory, "split_name": split_name,
            "name": os.path.basename(os.path.normpath(os.path.realpath(directory)))}


def _weights_of(run, checkpoint):
    from cifarbase.landscape import get_weights, load_checkpoint
    load_checkpoint(checkpoint["path"], run["model"], run["device"])
    return get_weights(run["model"]).clone()


def _mean_s(checkpoint):
    s = checkpoint.get("s", 1.0)
    return sum(s) / len(s) if isinstance(s, (list, tuple)) else float(s)


def _anchors(checkpoints):
    """Three checkpoints to span the continuation plane: low, middle, high s.

    Falls back to first / middle / last epoch when s never moved, so the figure
    still works on a baseline run -- there it shows the plane of the ordinary
    trajectory, which is the right comparison to draw against a homotopy one.
    """
    if len(checkpoints) < 3:
        return None
    ordered = sorted(checkpoints, key=_mean_s)
    if _mean_s(ordered[-1]) - _mean_s(ordered[0]) > 1e-6:
        middle = min(checkpoints,
                     key=lambda c: abs(_mean_s(c) - 0.5 * (_mean_s(ordered[0])
                                                           + _mean_s(ordered[-1]))))
        return [ordered[0], middle, ordered[-1]]
    return [checkpoints[0], checkpoints[len(checkpoints) // 2], checkpoints[-1]]


# --------------------------------------------------------------------------
# computing
# --------------------------------------------------------------------------

def _progress(label):
    def report(done, total, row):
        print(f"  {label}: row {done}/{total}, "
              f"loss {min(row):.3f}..{max(row):.3f}", flush=True)
    return report


def compute_surface(run, args):
    from cifarbase.landscape import (get_weights, random_direction, record,
                                     surface_2d)

    model = run["model"]
    center = get_weights(model).clone()
    e1 = random_direction(model, seed=args.seed, filter_norm=True)
    e2 = random_direction(model, seed=args.seed + 1, filter_norm=True)
    steps = [-args.span + 2 * args.span * i / (args.grid - 1)
             for i in range(args.grid)]

    out = surface_2d(model, run["split"], center, e1, e2, steps, steps,
                     batch_size=args.batch_size, max_images=args.max_images,
                     on_row=_progress("surface"))
    out.update(direction_seed=args.seed, span=args.span,
               max_images=args.max_images, split=run["split_name"],
               epoch=run["checkpoints"][-1].get("epoch"))
    record(run["dir"], out)
    return out


def compute_plane(run, args):
    """The continuation plane, with every checkpoint projected into it."""
    from cifarbase.landscape import (plane_from_points, project, record,
                                     surface_2d)

    anchors = _anchors(run["checkpoints"])
    if anchors is None:
        print("  plane: need at least 3 checkpoints, skipping")
        return None

    vectors = [_weights_of(run, c) for c in anchors]
    origin, e1, e2 = vectors[0], *plane_from_points(*vectors)

    # Project the whole trajectory first, so the grid can be sized to contain it
    # rather than to an arbitrary span: a plane plot whose contours stop before
    # the path does is worse than none.
    trajectory = []
    for checkpoint in run["checkpoints"]:
        x, y = project(_weights_of(run, checkpoint), origin, e1, e2)
        trajectory.append({"epoch": checkpoint.get("epoch"), "x": x, "y": y,
                           "s": _mean_s(checkpoint),
                           "test_acc": checkpoint.get("test_acc")})

    xs_seen = [p["x"] for p in trajectory]
    ys_seen = [p["y"] for p in trajectory]
    pad_x = 0.25 * (max(xs_seen) - min(xs_seen) + 1e-6)
    pad_y = 0.25 * (max(ys_seen) - min(ys_seen) + 1e-6)
    xs = [min(xs_seen) - pad_x + (max(xs_seen) - min(xs_seen) + 2 * pad_x)
          * i / (args.grid - 1) for i in range(args.grid)]
    ys = [min(ys_seen) - pad_y + (max(ys_seen) - min(ys_seen) + 2 * pad_y)
          * i / (args.grid - 1) for i in range(args.grid)]

    out = surface_2d(run["model"], run["split"], origin, e1, e2, xs, ys,
                     batch_size=args.batch_size, max_images=args.max_images,
                     on_row=_progress("plane"))
    out.update(kind="plane", trajectory=trajectory, split=run["split_name"],
               max_images=args.max_images,
               anchors=[{"epoch": c.get("epoch"), "s": _mean_s(c)}
                        for c in anchors])
    record(run["dir"], out)
    return out


def compute_s_alpha(run, args):
    from cifarbase.landscape import (get_weights, random_direction, record,
                                     surface_s_alpha)

    model = run["model"]
    center = get_weights(model).clone()
    direction = random_direction(model, seed=args.seed, filter_norm=True)
    s_values = [i / (args.s_grid - 1) for i in range(args.s_grid)]
    alphas = [-args.span + 2 * args.span * i / (args.grid - 1)
              for i in range(args.grid)]

    out = surface_s_alpha(model, run["split"], center, direction, s_values,
                          alphas, run["gate"], batch_size=args.batch_size,
                          max_images=args.max_images,
                          on_row=_progress("s_alpha"))
    out.update(direction_seed=args.seed, split=run["split_name"],
               max_images=args.max_images,
               epoch=run["checkpoints"][-1].get("epoch"))
    record(run["dir"], out)
    return out


def compute_loss_vs_s(run, args):
    """The homotopy slice at a handful of checkpoints along the trajectory."""
    from cifarbase.landscape import loss_vs_s, record

    s_values = [i / (args.s_grid - 1) for i in range(args.s_grid)]
    picks = _spread(run["checkpoints"], args.probes)
    rows = []
    for checkpoint in picks:
        _weights_of(run, checkpoint)
        out = loss_vs_s(run["model"], run["split"], s_values, run["gate"],
                        batch_size=args.batch_size, max_images=args.max_images)
        out.update(epoch=checkpoint.get("epoch"), s_trained=_mean_s(checkpoint))
        rows.append(out)
        print(f"  loss_vs_s: epoch {checkpoint.get('epoch')}, "
              f"loss {min(out['loss']):.3f}..{max(out['loss']):.3f}", flush=True)
    payload = {"kind": "loss_vs_s_series", "series": rows,
               "split": run["split_name"], "max_images": args.max_images}
    record(run["dir"], payload)
    return payload


def compute_ratio(run, args):
    """Residual ratio per block at every checkpoint, plus the gradient norms."""
    from cifarbase.landscape import grad_norm_per_block, record, residual_ratio

    x, y = next(run["split"].chunks(args.probe_batch))
    rows = []
    for checkpoint in run["checkpoints"]:
        _weights_of(run, checkpoint)
        s = checkpoint.get("s", 1.0)
        run["gate"].set(s if isinstance(s, (list, tuple)) else float(s))
        ratio = residual_ratio(run["model"], x, run["gate"])
        grads = grad_norm_per_block(run["model"], x, y, run["gate"])
        rows.append({"epoch": checkpoint.get("epoch"), "s": _mean_s(checkpoint),
                     "blocks": ratio["blocks"],
                     "grad_norm": [b["grad_norm"] for b in grads["blocks"]]})
    payload = {"kind": "ratio_series", "series": rows,
               "probe_batch": args.probe_batch}
    record(run["dir"], payload)
    return payload


def compute_curvature(run, args):
    from cifarbase.landscape import hutchinson_trace, record, top_hessian_eigs

    x, y = next(run["split"].chunks(args.probe_batch))
    rows = []
    for checkpoint in _spread(run["checkpoints"], args.probes):
        _weights_of(run, checkpoint)
        eigs = top_hessian_eigs(run["model"], x, y, k=2, iters=args.power_iters,
                                gate=run["gate"], s=1.0)
        trace = hutchinson_trace(run["model"], x, y, samples=args.hutchinson,
                                 gate=run["gate"], s=1.0)
        rows.append({"epoch": checkpoint.get("epoch"), "s": _mean_s(checkpoint),
                     "eigenvalues": eigs["eigenvalues"], "trace": trace["trace"],
                     "trace_sem": trace["sem"]})
        print(f"  curvature: epoch {checkpoint.get('epoch')}, "
              f"top eig {eigs['eigenvalues'][0]:.2f}, "
              f"trace {trace['trace']:.1f}", flush=True)
    payload = {"kind": "curvature_series", "series": rows,
               "probe_batch": args.probe_batch,
               # Measured with the gate forced to 1 so curvature is compared at
               # the same point of the homotopy across every checkpoint.
               "measured_at_s": 1.0}
    record(run["dir"], payload)
    return payload


def compute_barrier(run, other_dir, args):
    """Interpolate to another run's solution, recomputing BatchNorm throughout."""
    from cifarbase.landscape import get_weights, interpolate, record

    other = prepare(other_dir, run["split_name"], seed=None)
    w_b = get_weights(other["model"]).clone().to(run["device"])
    del other

    _weights_of(run, run["checkpoints"][-1])
    w_a = get_weights(run["model"]).clone()

    out = interpolate(run["model"], run["split"], w_a, w_b, n=args.barrier_points,
                      extend=0.1, train_split=run["train"],
                      bn_batches=args.bn_batches, batch_size=args.batch_size,
                      max_images=args.max_images)
    out.update(against=os.path.basename(os.path.normpath(other_dir)),
               split=run["split_name"], max_images=args.max_images)
    record(run["dir"], out)
    print(f"  barrier: {out['barrier']:.4f} "
          f"({'connected' if out['barrier'] < 0.02 else 'NOT connected'})")
    return out


def _spread(items, count):
    """`count` items spread evenly across a list, endpoints always included."""
    if count >= len(items):
        return list(items)
    step = (len(items) - 1) / max(1, count - 1)
    return [items[round(i * step)] for i in range(count)]


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def _levels(grid, count=18, decades=3.0):
    """Log-spaced contour levels, floored relative to the data's own range.

    Linear levels spend every line inside the basin, where the loss barely
    changes, and leave the walls -- the part that shows the shape -- blank. But
    log levels anchored at an absolute floor are just as bad the other way: a
    trained network has a minimum near 0.01 and walls near 5, so a floor of 1e-4
    would put half the contours in a range the surface never visits.

    Hence a floor three decades under the peak. Falls back to linear levels when
    the minimum is not positive, which happens on a surface that dips below zero
    only because it is being drawn from something other than a loss.
    """
    import numpy as np

    values = np.asarray(grid, dtype=float)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return count
    if low <= 0:
        return np.linspace(low, high, count)
    return np.geomspace(max(low, high / 10 ** decades), high, count)


def _contour(ax, xs, ys, grid, label="loss", pad=0.02):
    import numpy as np

    values = np.asarray(grid, dtype=float)
    levels = _levels(values)
    filled = ax.contourf(xs, ys, values, levels=levels, cmap=CMAP, extend="both")
    ax.contour(xs, ys, values, levels=levels, colors="#0b0b0b", linewidths=0.35,
               alpha=0.45)
    bar = ax.figure.colorbar(filled, ax=ax, pad=pad, fraction=0.046)
    bar.set_label(label, fontsize=8)
    bar.ax.tick_params(labelsize=7)
    return filled


def plot_surface(record_, run_name, outdir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    _contour(ax, record_["xs"], record_["ys"], record_["loss"])
    ax.plot(0, 0, marker="*", ms=14, color="#ffffff", markeredgecolor="#0b0b0b",
            markeredgewidth=0.8, zorder=5)
    ax.set_xlabel("filter-normalised direction 1")
    ax.set_ylabel("filter-normalised direction 2")
    ax.set_title(f"{run_name}  --  loss surface at the final weights\n"
                 f"{record_.get('split', '?')} loss on "
                 f"{record_.get('max_images', '?')} images, "
                 f"directions seeded {record_.get('direction_seed')}")
    return _save(fig, outdir, "surface.png")


def plot_plane(record_, run_name, outdir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    path = record_["trajectory"]
    xs = [p["x"] for p in path]
    ys = [p["y"] for p in path]
    scatter = ax.scatter(xs, ys, c=[p["s"] for p in path], cmap="autumn_r",
                         s=42, zorder=5, edgecolors="#0b0b0b", linewidths=0.6,
                         vmin=0.0, vmax=1.0)
    # Horizontal, and underneath. Two vertical colorbars on one axes fit side by
    # side but their labels do not: they end up stacked in the same gutter and
    # both become unreadable. Different orientations cannot collide.
    bar = fig.colorbar(scatter, ax=ax, orientation="horizontal", location="bottom",
                       pad=0.13, fraction=0.05, shrink=0.75)
    bar.set_label("s at that checkpoint", fontsize=8)
    bar.ax.tick_params(labelsize=7)

    _contour(ax, record_["xs"], record_["ys"], record_["loss"])
    ax.plot(xs, ys, color="#ffffff", lw=2.4, alpha=0.85, zorder=4)
    ax.scatter(xs, ys, c=[p["s"] for p in path], cmap="autumn_r", s=42,
               zorder=6, edgecolors="#0b0b0b", linewidths=0.6, vmin=0.0,
               vmax=1.0)

    for point in (path[0], path[-1]):
        ax.annotate(f"ep {point['epoch']}", (point["x"], point["y"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7,
                    color="#ffffff")

    anchors = record_.get("anchors", [])
    detail = (", ".join(f"ep {a['epoch']} (s={a['s']:.2f})" for a in anchors)
              if anchors else "")
    ax.set_xlabel("towards the second anchor")
    ax.set_ylabel("orthogonal component of the third")
    ax.set_title(f"{run_name}  --  the continuation plane, trajectory projected\n"
                 f"plane through {detail}")
    return _save(fig, outdir, "plane.png")


def plot_s_alpha(record_, run_name, outdir):
    import matplotlib.pyplot as plt
    import numpy as np

    values = np.asarray(record_["loss"], dtype=float)      # [alpha, s]
    s_values = np.asarray(record_["s_values"], dtype=float)
    alphas = np.asarray(record_["alphas"], dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    _contour(ax, s_values, alphas, values)

    # The valley floor: for each s, where along the weight direction the loss is
    # lowest. If it drifts, the minimum is moving as the residual branches switch
    # on, which is the whole premise of tracking theta*(s) rather than jumping to
    # s=1 and optimising there.
    floor = alphas[values.argmin(axis=0)]
    ax.plot(s_values, floor, color="#ffffff", lw=1.8, ls="--",
            label="argmin over alpha")
    ax.axhline(0.0, color="#0b0b0b", lw=0.8, alpha=0.6)
    ax.legend(loc="upper right", labelcolor="#ffffff")

    ax.set_xlabel("homotopy parameter s")
    ax.set_ylabel("alpha  (displacement along a filter-normalised direction)")
    ax.set_title(f"{run_name}  --  how the landscape deforms as s rises\n"
                 f"L_s(theta + alpha*d), {record_.get('split', '?')} loss on "
                 f"{record_.get('max_images', '?')} images")
    return _save(fig, outdir, "s_alpha.png")


def plot_loss_vs_s(record_, run_name, outdir):
    import matplotlib.pyplot as plt

    series = record_["series"]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for i, row in enumerate(series):
        colour = CYCLE[i % len(CYCLE)]
        label = f"ep {row['epoch']} (trained at s={row['s_trained']:.2f})"
        ax_loss.plot(row["s_values"], row["loss"], color=colour, lw=1.6,
                     label=label)
        ax_acc.plot(row["s_values"], row["acc"], color=colour, lw=1.6)
        # Where this checkpoint's weights were actually trained: the distance
        # between the marker and the curve's minimum is the interesting part.
        ax_loss.axvline(row["s_trained"], color=colour, lw=0.7, ls=":", alpha=0.6)

    ax_loss.set_ylabel("loss")
    ax_loss.set_title("L_s(theta) at fixed weights")
    ax_loss.legend(fontsize=7)
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("accuracy over the same sweep")
    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("homotopy parameter s")
    fig.suptitle(f"{run_name}  --  reading the same weights all along the "
                 f"homotopy", x=0.5, y=1.02, fontsize=11)
    return _save(fig, outdir, "loss_vs_s.png")


def plot_ratio(record_, run_name, outdir):
    import matplotlib.pyplot as plt
    import numpy as np

    series = record_["series"]
    epochs = [row["epoch"] for row in series]
    ratios = np.asarray([[b["ratio"] for b in row["blocks"]] for row in series])
    raw = np.asarray([[b["f_rms"] / (b["skip_rms"] + 1e-10)
                       for b in row["blocks"]] for row in series])

    fig, (ax_r, ax_raw, ax_s) = plt.subplots(
        1, 3, figsize=(13.5, 4.0),
        gridspec_kw={"width_ratios": [1, 1, 0.8], "wspace": 0.28})

    for ax, values, title in (
            (ax_r, ratios, "rms(s*F) / rms(shortcut)"),
            (ax_raw, raw, "the same with s divided out")):
        mesh = ax.pcolormesh(epochs, range(values.shape[1]), values.T,
                             cmap=CMAP, shading="nearest")
        fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046).ax.tick_params(
            labelsize=7)
        ax.set_xlabel("epoch")
        ax.set_ylabel("block (forward order)")
        ax.set_title(title)

    for i in range(ratios.shape[1]):
        ax_s.plot(epochs, ratios[:, i], color=CYCLE[i % len(CYCLE)], lw=1.3,
                  label=f"block {i}")
    ax_s.set_xlabel("epoch")
    ax_s.set_ylabel("ratio")
    ax_s.set_title("per block")
    ax_s.legend(fontsize=6, ncol=2)

    # The right-hand panel is the one that answers the question: if s is being
    # absorbed by gamma_bn2, the middle panel rises exactly as fast as s does and
    # the left panel stays flat.
    fig.suptitle(f"{run_name}  --  how loud each residual branch is, and whether "
                 f"s is doing the work", x=0.5, y=1.03, fontsize=11)
    return _save(fig, outdir, "ratio.png")


def plot_curvature(record_, run_name, outdir):
    import matplotlib.pyplot as plt

    series = record_["series"]
    epochs = [row["epoch"] for row in series]
    fig, (ax_eig, ax_trace) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    for index, colour, label in ((0, TEST, "top eigenvalue"),
                                 (1, MUTED, "second")):
        values = [row["eigenvalues"][index] if len(row["eigenvalues"]) > index
                  else float("nan") for row in series]
        ax_eig.plot(epochs, values, color=colour, lw=1.6, marker="o", ms=3.5,
                    label=label)
    ax_eig.set_ylabel("eigenvalue")
    ax_eig.set_title("Hessian, largest in magnitude")
    ax_eig.legend()

    traces = [row["trace"] for row in series]
    errors = [2 * row["trace_sem"] for row in series]
    ax_trace.errorbar(epochs, traces, yerr=errors, color=TRAIN, lw=1.6,
                      marker="o", ms=3.5, capsize=3)
    ax_trace.set_ylabel("trace")
    ax_trace.set_title("Hessian trace (Hutchinson, +-2 sem)")

    for ax in (ax_eig, ax_trace):
        ax.set_xlabel("epoch")
        ax.axhline(0, color="#0b0b0b", lw=0.8, alpha=0.5)
    fig.suptitle(f"{run_name}  --  curvature along the trajectory, all measured "
                 f"at s=1 on {record_.get('probe_batch')} images",
                 x=0.5, y=1.03, fontsize=11)
    return _save(fig, outdir, "curvature.png")


def plot_barrier(records, run_name, outdir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, row in enumerate(records):
        colour = CYCLE[i % len(CYCLE)]
        ax.plot(row["alphas"], row["loss"], color=colour, lw=1.7,
                label=f"-> {row.get('against', '?')}  "
                      f"(barrier {row['barrier']:.3f})")
    ax.axvline(0.0, color=MUTED, lw=0.8)
    ax.axvline(1.0, color=MUTED, lw=0.8)
    ax.set_xlabel("alpha  (0 = this run's solution, 1 = the other's)")
    ax.set_ylabel("loss")
    ax.legend(fontsize=7)
    recomputed = all(r.get("bn_recomputed") for r in records)
    ax.set_title(f"{run_name}  --  linear interpolation between solutions\n"
                 f"BatchNorm {'recomputed at every point' if recomputed else 'NOT recomputed -- barrier is unreliable'}")
    return _save(fig, outdir, "barrier.png")


def _save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def draw_all(directory, outdir, run_name):
    """Redraw every figure the run's landscape.jsonl has data for.

    Later records of a kind win, so recomputing one figure and redrawing picks up
    the new version without having to clear the file.
    """
    from cifarbase.landscape import read_records

    rows = read_records(directory)
    if not rows:
        print(f"no measurements in {directory}/landscape.jsonl")
        return []

    latest = {}
    barriers = []
    for row in rows:
        if row.get("kind") == "interpolate":
            barriers.append(row)
        else:
            latest[row.get("kind")] = row

    made = []
    drawers = [("surface_2d", plot_surface), ("plane", plot_plane),
               ("surface_s_alpha", plot_s_alpha),
               ("loss_vs_s_series", plot_loss_vs_s), ("ratio_series", plot_ratio),
               ("curvature_series", plot_curvature)]
    for kind, drawer in drawers:
        if kind in latest:
            made.append(drawer(latest[kind], run_name, outdir))
    if barriers:
        made.append(plot_barrier(barriers, run_name, outdir))
    return made


def main():
    parser = argparse.ArgumentParser(
        description="Draw the loss landscape of a run trained with ckpt_every > 0.")
    parser.add_argument("folder", help="a run directory made by main.py")
    parser.add_argument("--only", default=None,
                        help=f"comma-separated subset of {', '.join(FIGURES)}")
    parser.add_argument("--against", default=None, metavar="RUN",
                        help="another run directory, for the interpolation barrier")
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw from landscape.jsonl without computing")
    parser.add_argument("--split", default="train", choices=("train", "test"),
                        help="which split the loss is measured on (default: train, "
                             "the objective that was actually minimised)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the random directions (default: 0)")
    parser.add_argument("--grid", type=int, default=21,
                        help="points per axis on a surface (default: 21)")
    parser.add_argument("--s-grid", type=int, default=11,
                        help="points along the s axis (default: 11)")
    parser.add_argument("--span", type=float, default=1.0,
                        help="half-width of a surface, in filter-normalised "
                             "units (default: 1.0)")
    parser.add_argument("--max-images", type=int, default=5000,
                        help="images per loss evaluation (default: 5000)")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--probes", type=int, default=6,
                        help="checkpoints to measure for the per-checkpoint "
                             "figures (default: 6)")
    parser.add_argument("--probe-batch", type=int, default=512,
                        help="batch size for curvature and ratios (default: 512)")
    parser.add_argument("--power-iters", type=int, default=30)
    parser.add_argument("--hutchinson", type=int, default=12)
    parser.add_argument("--barrier-points", type=int, default=13)
    parser.add_argument("--bn-batches", type=int, default=64,
                        help="batches used to re-estimate BatchNorm at each "
                             "interpolation point (default: 64)")
    parser.add_argument("--out", default=None,
                        help="where the pngs go (default: <folder>/figures)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise SystemExit(f"!! not a directory: {args.folder}")

    wanted = FIGURES if args.only is None else tuple(
        piece.strip() for piece in args.only.split(","))
    unknown = sorted(set(wanted) - set(FIGURES))
    if unknown:
        raise SystemExit(f"!! unknown figure(s) {', '.join(unknown)}: "
                         f"pick from {', '.join(FIGURES)}")

    name = os.path.basename(os.path.normpath(os.path.realpath(args.folder)))

    if not args.plot_only:
        run = prepare(args.folder, args.split)
        print(f"{name}: {len(run['checkpoints'])} checkpoints for seed "
              f"{run['seed']}, epochs "
              f"{run['checkpoints'][0].get('epoch')}.."
              f"{run['checkpoints'][-1].get('epoch')}")
        steps = [("ratio", compute_ratio), ("loss_vs_s", compute_loss_vs_s),
                 ("curvature", compute_curvature), ("surface", compute_surface),
                 ("plane", compute_plane), ("s_alpha", compute_s_alpha)]
        # Cheapest first, so an interrupted sweep still leaves the diagnostics
        # that matter most per second spent.
        for key, step in steps:
            if key in wanted:
                print(f"computing {key}...")
                step(run, args)
        if "barrier" in wanted and args.against:
            print("computing barrier...")
            compute_barrier(run, args.against, args)
        elif "barrier" in wanted and args.only:
            print("  barrier: needs --against RUN, skipping")

    import matplotlib
    matplotlib.use("Agg")           # no display on Kaggle, and none needed here
    _theme(matplotlib)

    outdir = args.out or os.path.join(args.folder, "figures")
    os.makedirs(outdir, exist_ok=True)
    made = draw_all(args.folder, outdir, name)

    print()
    for path in made:
        if path:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()

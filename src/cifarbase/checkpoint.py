"""Save and restore a run, including the things that make the anchor meaningful.

The failure this file exists to prevent: a resume that rebuilds the model from a
fresh seed, or recomputes K_0 at w_k instead of w_0. Either produces a run that
trains fine, logs a plausible d_t, and answers a question nobody asked -- the
anchor is now pointing at wherever the run happened to be when it was
interrupted, and d_t is measured against that too, so both curves look normal.
Nothing downstream can detect it.

So everything the anchor and the controller depend on is written together, in one
file, or not at all:

    w0            in the model state_dict, as buffers under _anchor
    K_0           the reference sketches, verbatim
    probe set     the indices, and the tangent seed they were drawn with
    controller    u, d_ema, the monitor windows, the clip count
    allocator     a_g, the tension EMAs, the slow-loop probe counter
    RNG           all four streams, so the batch order continues rather than restarts

torch.save of a dict rather than anything cleverer: this has to be readable by a
`python -c` six months from now with no project code on the path.
"""
import os

import torch

# Bumped from 1: `reference` gained the per-group sketches, which are a nested
# dict, and `allocator` is new. A format-1 checkpoint has neither and resuming one
# into this build would restart the allocation from uniform without saying so.
FORMAT = 2


def _to_cpu(value):
    """Sketches to CPU, through the one level of nesting reference now has."""
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    return value.detach().to("cpu")


def _to_device(value, device):
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value.to(device)


def save(path, *, cfg, seed, epoch, step, model, optimizer, controller,
         allocator, reference, probe_index, history, probes, generator):
    """Write one checkpoint atomically. Returns the path."""
    payload = {
        "format": FORMAT,
        "config": cfg,
        "seed": seed,
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "controller": controller.state_dict(),
        "allocator": allocator.state_dict(),
        # float64 on the CPU: the reference sketches are what every later d_t is
        # measured against, so a lossy round-trip here would move the whole curve.
        # None is legal: a checkpoint written before the reference step has no K_0
        # yet, and inventing one at w_k is exactly the corruption this file exists
        # to prevent.
        "reference": (None if reference is None else _to_cpu(reference)),
        "probe_index": probe_index.detach().to("cpu"),
        "history": history,
        "probes": probes,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
            # The batch-order stream is a LOCAL generator, deliberately separate
            # from the global one so the probe can never perturb it. Being local
            # also means torch.get_rng_state() does not cover it, and a resume
            # that forgot it would replay epoch 0's shuffle from epoch 40.
            "data": generator.get_state(),
        },
    }
    partial = path + ".part"
    torch.save(payload, partial)
    os.replace(partial, path)
    return path


def load(path, model, optimizer, controller, anchor, device, generator, cfg,
         *, allocator=None):
    """Restore in place and return what the training loop needs to continue.

    The shape assertion runs after the weights land: a checkpoint from a
    different arch or width would otherwise be pulled toward a w0 of the wrong
    shape, which fails somewhere unrelated several minutes later.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != FORMAT:
        raise SystemExit(f"!! {path} is checkpoint format "
                         f"{payload.get('format')!r}, this build reads {FORMAT}")

    # The lr schedule is a function of total_steps = steps_per_epoch * epochs, so
    # resuming into a run with a different `epochs` silently puts the second half
    # of training on a different cosine curve than the first. Nothing downstream
    # can see that: the loss keeps falling and the lr column looks like a valid
    # schedule, just not the one the checkpoint was produced on. Caught here
    # rather than left to be discovered in a comparison plot.
    # anchor_grouping is in this list for a different reason than the rest: the
    # allocator's a_g and tension EMAs are keyed BY GROUP NAME, so a resume that
    # changed the partition would load an allocation belonging to groups that no
    # longer exist -- silently, since load_state_dict just takes the dict.
    for key in ("epochs", "batch_size", "arch", "width", "schedule",
                "warmup_epochs", "lr", "train_subset", "anchor_grouping"):
        before, now = payload["config"].get(key), cfg.get(key)
        if before != now:
            raise SystemExit(
                f"!! {path} was written with {key}={before!r} but this run has "
                f"{key}={now!r}. Resuming would continue a different training "
                f"run: total_steps, and therefore the whole lr curve, is derived "
                f"from these. To stop a run early and continue it later, keep "
                f"every one of them fixed and use --stop-after-epoch.")

    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    controller.load_state_dict(payload["controller"])
    if allocator is not None:
        allocator.load_state_dict(payload["allocator"])
    anchor.assert_shapes()

    torch.set_rng_state(payload["rng"]["torch"].to("cpu", torch.uint8))
    if payload["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [state.to("cpu", torch.uint8) for state in payload["rng"]["cuda"]])
    generator.set_state(payload["rng"]["data"].to("cpu", torch.uint8))

    reference = (None if payload["reference"] is None else
                 _to_device(payload["reference"], device))
    print(f"resumed from {path}: seed {payload['seed']}, epoch "
          f"{payload['epoch']}, step {payload['step']}, "
          f"lambda {controller.lam(payload['step']):.4g}")
    return {"seed": payload["seed"], "epoch": payload["epoch"],
            "step": payload["step"], "reference": reference,
            "probe_index": payload["probe_index"].to(device),
            "history": payload["history"], "probes": payload["probes"],
            "config": payload["config"]}

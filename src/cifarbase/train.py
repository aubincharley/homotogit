"""One training run, from a seed to a row of numbers.

Two accuracies get reported and they answer different questions:

  test_acc_final     the last epoch. What you get if you just train and stop.
  test_acc_selected  the epoch with the best validation accuracy. What you get
                     with honest model selection.

A baseline has to state which one it means, so it reports both and never mixes
them. The reference config runs with val_size=0: there is nothing to select on,
the two coincide, and the report says `selection: final_epoch` rather than
pretending a selection happened.
"""
import math
import time

import torch
import torch.nn.functional as F

from cifarbase.homotopy import (ResidualGate, other_parameters, phase_at,
                                phase_bounds, residual_parameters, s_at_cfg)
from cifarbase.landscape import (grad_norm_per_block, record, residual_ratio,
                                 save_checkpoint)
from cifarbase.metrics import evaluate
from cifarbase.model import build_model, count_params
from cifarbase.utils.seeding import seed_everything


def build_optimizer(model, cfg):
    """SGD, and only SGD.

    Keeping the baseline to one optimiser is the point of the repo: a number you
    can compare against needs a recipe that did not drift. Adding AdamW here is a
    two-line change, and should come with its own config file and its own number.
    """
    if cfg["optimizer"] != "sgd":
        raise ValueError(f"unknown optimizer {cfg['optimizer']!r}: this baseline "
                         f"is sgd only")
    shared = dict(lr=cfg["lr"], momentum=cfg["momentum"],
                  weight_decay=cfg["weight_decay"],
                  nesterov=bool(cfg["nesterov"]) and cfg["momentum"] > 0)
    if not cfg["lr_gate_control"]:
        return torch.optim.SGD(model.parameters(), **shared)

    # The control arm. Because dL/dtheta_F is proportional to s, the homotopy is
    # partly just a branch-local learning-rate schedule; this splits the
    # parameters along exactly the line s acts on, so the same profile can be
    # applied to the lr with the forward pass left alone. The shortcut
    # projections stay in the ungated group -- s never touches them either.
    return torch.optim.SGD([
        {"params": other_parameters(model), "gated": False},
        {"params": residual_parameters(model), "gated": True},
    ], **shared)


def lr_at(step, total_steps, cfg, bounds=()):
    """Multiplier on the base lr. Warmup is linear, then the chosen schedule.

    Computed per step rather than per epoch so the curve does not depend on how
    many batches an epoch happens to contain -- otherwise changing batch_size or
    train_subset silently changes the schedule too, and the comparison you were
    trying to make is gone.

    With `bounds`, the same curve is run inside each continuation phase instead
    of once across the whole run: each plateau of s gets its own warmup and its
    own decay, so it can actually converge before s steps. Warmup keeps the same
    *fraction* of a phase that warmup_epochs is of the run, which is what makes
    a restarted schedule comparable to the global one rather than a new
    hyperparameter. Empty bounds leave every existing arm untouched.
    """
    if bounds:
        index, local = phase_at(step / max(1, total_steps), bounds)
        total_steps = max(1, round((bounds[index + 1] - bounds[index]) * total_steps))
        step = min(total_steps - 1, int(local * total_steps))
    warmup = int(cfg["warmup_epochs"] * total_steps / max(cfg["epochs"], 1))
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    if cfg["schedule"] == "cosine":
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    if cfg["schedule"] == "step":
        return 0.1 ** sum(progress >= milestone for milestone in (0.5, 0.75))
    return 1.0


def diagnose(model, gate, probe, out_dir, seed, epoch, s_values):
    """Per-block residual ratio and gradient norm, written out mid-run.

    Two numbers per block, and between them they say whether the homotopy is
    doing anything:

      ratio      rms(s*F) / rms(shortcut) -- how loud the branch actually is
      grad_norm  what reaches the branch's parameters; exactly 0 at s=0

    The ratio is the one to watch. Because F ends in bn2, the network can cancel
    a small s by growing gamma, and if it does, the ratio climbs on its own
    schedule rather than following s -- meaning the homotopy was reparametrised
    away and the run is a baseline wearing a costume. Waiting until the run
    finishes to discover that costs a day per arm.

    Measured on one fixed batch, so the series across epochs is comparable.
    """
    x, y = probe
    ratio = residual_ratio(model, x, gate)
    grads = grad_norm_per_block(model, x, y, gate)
    payload = {"kind": "train_diagnostics", "seed": seed, "epoch": epoch,
               "s": list(s_values),
               "ratio": [b["ratio"] for b in ratio["blocks"]],
               "f_rms": [b["f_rms"] for b in ratio["blocks"]],
               "skip_rms": [b["skip_rms"] for b in ratio["blocks"]],
               "grad_norm": [b["grad_norm"] for b in grads["blocks"]],
               "grad_norm_total": grads["total"]}
    if out_dir:
        record(out_dir, payload)
    return payload


def train_once(cfg, train, val, test, device, seed, run=None, verbose=True,
               out_dir=None):
    seed_everything(seed, cfg["deterministic"])
    model = build_model(cfg, device, quiet=not verbose)
    optimizer = build_optimizer(model, cfg)
    gate = ResidualGate(model)

    # batches() drops the last partial batch, so a split shorter than one batch
    # yields nothing at all and the epoch would divide by zero a long way from
    # here. This bites when train_subset is set small for debugging.
    if len(train) < cfg["batch_size"]:
        raise SystemExit(f"!! train split has {len(train)} examples but "
                         f"batch_size is {cfg['batch_size']}: no full batch. "
                         f"Raise train_subset or lower batch_size.")

    steps_per_epoch = max(1, len(train) // cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["epochs"]
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    eval_bs = cfg["eval_batch_size"]

    generator = torch.Generator(device=device).manual_seed(seed)
    best = {"val_acc": -1.0, "epoch": -1, "state": None}
    history = []
    step = 0
    images = 0
    s_values = gate.get()
    started = time.time()

    # One fixed, un-augmented batch, drawn once. Diagnostics measured on a
    # different batch every epoch would move for reasons that have nothing to do
    # with training, and the series is the whole point of measuring them.
    probe = next(train.chunks(cfg["batch_size"])) if cfg["diag_every"] else None

    bounds = phase_bounds(cfg)
    previous_phase = None
    if bounds and verbose:
        print(f"  [seed {seed}] continuation: {len(bounds) - 1} phases, lr "
              f"restarts at each -- boundaries "
              f"{', '.join(f'{b:.2f}' for b in bounds)}")

    # theta at initialisation is the natural origin for a trajectory plot, and
    # it is the one checkpoint that cannot be recovered afterwards.
    if out_dir and cfg["ckpt_every"]:
        save_checkpoint(model, out_dir, f"seed{seed}_init",
                        meta={"seed": seed, "epoch": -1, "s": s_values})

    model.train()

    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        running = correct = seen = 0
        for x, y in train.batches(cfg["batch_size"], generator,
                                  augment=cfg["augment"]):
            # Per step rather than per epoch for the same reason lr_at is: the
            # schedule must not depend on how many batches an epoch contains.
            progress = (epoch / cfg["epochs"] if cfg["s_granularity"] == "epoch"
                        else step / max(1, total_steps))
            s_values = s_at_cfg(progress, len(gate), cfg)

            # The control arm trains the real ResNet and moves the learning rate
            # instead, so its forward pass stays at s=1 throughout.
            gate.set(1.0 if cfg["lr_gate_control"] else s_values)
            gated_scale = (sum(s_values) / len(s_values)
                           if cfg["lr_gate_control"] else 1.0)

            for group, base in zip(optimizer.param_groups, base_lrs, strict=True):
                scale = lr_at(step, total_steps, cfg, bounds)
                if group.get("gated"):
                    scale *= gated_scale
                group["lr"] = base * scale

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y,
                                   label_smoothing=cfg["label_smoothing"])
            loss.backward()
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

            running += float(loss.detach()) * y.numel()
            correct += int((logits.detach().argmax(1) == y).sum())
            seen += y.numel()
            images += y.numel()
            step += 1

        epoch_s = time.time() - epoch_start
        row = {"seed": seed, "epoch": epoch,
               "lr": optimizer.param_groups[0]["lr"],
               "train_loss": running / seen, "train_acc_batchwise": correct / seen,
               "epoch_s": epoch_s, "img_per_s": seen / epoch_s,
               "s_mean": sum(s_values) / len(s_values),
               "s_min_block": min(s_values), "s_max_block": max(s_values),
               # The whole vector, not just its spread: under `sequential` the
               # blocks hold different values all through the ramp, and a mean
               # would describe none of them.
               "s_blocks": list(s_values),
               "phase": phase_at(progress, bounds)[0] if bounds else 0}

        if val is not None:
            val_metrics = evaluate(model, val, eval_bs)
            row["val_loss"] = val_metrics["loss"]
            row["val_acc"] = val_metrics["acc"]
            if val_metrics["acc"] > best["val_acc"]:
                # Kept on CPU: 11M params is 45 MB, and a GPU copy on every
                # improvement is needless pressure on a card already holding the
                # whole dataset.
                best = {"val_acc": val_metrics["acc"], "epoch": epoch,
                        "state": {k: v.detach().to("cpu", copy=True)
                                  for k, v in model.state_dict().items()}}

        test_metrics = evaluate(model, test, eval_bs)
        row["test_loss"] = test_metrics["loss"]
        row["test_acc"] = test_metrics["acc"]

        # Two different questions, and mixing them up is how a homotopy run gets
        # compared against a baseline it never matched: test_acc is the model as
        # it currently runs, at whatever s the schedule is holding; the s=1
        # readout is the ResNet you would have if you stopped here. Only the
        # second is comparable to a baseline number. Skipped when s is already 1
        # so the baseline path costs nothing extra.
        if min(s_values) == max(s_values) == 1.0:
            row["test_acc_at_s1"] = row["test_acc"]
        else:
            with gate.at(1.0):
                row["test_acc_at_s1"] = evaluate(model, test, eval_bs)["acc"]
        if cfg["diag_every"] and (epoch % cfg["diag_every"] == 0
                                  or epoch == cfg["epochs"] - 1):
            diagnostics = diagnose(model, gate, probe, out_dir, seed, epoch,
                                   s_values)
            # Summaries into the epoch row as well, so the two numbers that
            # matter show up in history.csv without anyone having to open the
            # jsonl. The per-block detail stays in landscape.jsonl.
            blocks = diagnostics["ratio"]
            row["ratio_mean"] = sum(blocks) / len(blocks)
            row["ratio_last_block"] = blocks[-1]
            row["grad_norm_mean"] = (sum(diagnostics["grad_norm"])
                                     / len(diagnostics["grad_norm"]))
            row["grad_norm_total"] = diagnostics["grad_norm_total"]
            model.train()               # diagnose() evaluates, so put it back

        history.append(row)

        if out_dir and cfg["ckpt_every"] and (
                epoch % cfg["ckpt_every"] == 0 or epoch == cfg["epochs"] - 1):
            save_checkpoint(model, out_dir, f"seed{seed}_epoch{epoch:03d}",
                            meta={"seed": seed, "epoch": epoch, "s": s_values,
                                  "test_acc": row["test_acc"]})

        if run is not None:
            run.log({f"s{seed}/{k}": v for k, v in row.items() if k != "seed"})
        if verbose and bounds and row["phase"] != previous_phase:
            # The moment the continuation steps to a new s and the lr restarts.
            # Without this the restart is invisible in the log and the run looks
            # like a schedule that broke.
            settled = (f", grad norm {row['grad_norm_total']:.3f} leaving the "
                       f"last phase" if "grad_norm_total" in row else "")
            print(f"  [seed {seed}] --- phase {row['phase']} begins, "
                  f"s={row['s_mean']:.3f}{settled}")
            previous_phase = row["phase"]
        if verbose:
            val_part = f"val={row['val_acc']:.4f}  " if val is not None else ""
            s_part = (f"s={row['s_mean']:.2f}  "
                      if row["s_mean"] != 1.0 or cfg["s_schedule"] != "const"
                      else "")
            print(f"  [seed {seed}] epoch {epoch:03d}  lr={row['lr']:.4f}  "
                  f"{s_part}loss={row['train_loss']:.4f}  {val_part}"
                  f"test={row['test_acc']:.4f}  {row['epoch_s']:.1f}s")

    wall = time.time() - started
    final = _describe(model, cfg, train, test)

    # Re-measure on the selected checkpoint rather than reusing the epoch row:
    # the row only has test loss and accuracy, and reporting a selected accuracy
    # next to final-epoch per-class numbers would be mixing two models.
    if best["state"] is not None:
        live = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best["state"])
        selected = _describe(model, cfg, train, test)
        model.load_state_dict(live)
    else:
        selected = dict(final)

    summary = {
        "seed": seed, "params": count_params(model),
        "epochs": cfg["epochs"], "wall_s": wall,
        "img_per_s": images / wall,
        "selection": "best_val" if best["state"] is not None else "final_epoch",
        "selected_epoch": (best["epoch"] if best["state"] is not None
                           else cfg["epochs"] - 1),
        "best_val_acc": (best["val_acc"] if best["state"] is not None
                         else float("nan")),
        "test_acc_final": final["test_acc"], "test_loss_final": final["test_loss"],
        "test_acc_selected": selected["test_acc"],
        "test_loss_selected": selected["test_loss"],
        "test_err_selected": 100.0 * (1.0 - selected["test_acc"]),
        "train_acc": final["train_acc"], "train_loss": final["train_loss"],
        "gen_gap": final["train_acc"] - final["test_acc"],
        "worst_class": final["worst_class"],
        "worst_class_acc": final["worst_class_acc"],
        "per_class": final.get("per_class"),
        # Recorded so a results.json can be read years later without having to
        # reconstruct what the schedule was doing when the number was taken.
        "s_schedule": cfg["s_schedule"],
        "s_final": sum(s_values) / len(s_values),
    }
    gate.close()
    if verbose:
        print(f"  [seed {seed}] {wall:.0f}s  "
              f"test_final={summary['test_acc_final']:.4f}  "
              f"test_selected={summary['test_acc_selected']:.4f}  "
              f"train={summary['train_acc']:.4f}  "
              f"gap={summary['gen_gap']:+.4f}")
    if run is not None:
        run.log({f"summary/s{seed}/{k}": v for k, v in summary.items()
                 if k not in ("per_class",)})
    return summary, history


def _describe(model, cfg, train, test):
    """Every end-of-run metric for whatever weights are currently loaded."""
    eval_bs = cfg["eval_batch_size"]
    test_metrics = evaluate(model, test, eval_bs, per_class=cfg["per_class"])
    train_metrics = evaluate(model, train, eval_bs)
    return {
        "test_acc": test_metrics["acc"], "test_loss": test_metrics["loss"],
        "train_acc": train_metrics["acc"], "train_loss": train_metrics["loss"],
        "worst_class": test_metrics.get("worst_class", -1),
        "worst_class_acc": test_metrics.get("worst_class_acc", float("nan")),
        "per_class": test_metrics.get("per_class"),
    }

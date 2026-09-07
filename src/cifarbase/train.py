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
    return torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                           momentum=cfg["momentum"],
                           weight_decay=cfg["weight_decay"],
                           nesterov=bool(cfg["nesterov"]) and cfg["momentum"] > 0)


def lr_at(step, total_steps, cfg):
    """Multiplier on the base lr. Warmup is linear, then the chosen schedule.

    Computed per step rather than per epoch so the curve does not depend on how
    many batches an epoch happens to contain -- otherwise changing batch_size or
    train_subset silently changes the schedule too, and the comparison you were
    trying to make is gone.
    """
    warmup = int(cfg["warmup_epochs"] * total_steps / max(cfg["epochs"], 1))
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    if cfg["schedule"] == "cosine":
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    if cfg["schedule"] == "step":
        return 0.1 ** sum(progress >= milestone for milestone in (0.5, 0.75))
    return 1.0


def train_once(cfg, train, val, test, device, seed, run=None, verbose=True):
    seed_everything(seed, cfg["deterministic"])
    model = build_model(cfg, device, quiet=not verbose)
    optimizer = build_optimizer(model, cfg)

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
    started = time.time()
    model.train()

    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        running = correct = seen = 0
        for x, y in train.batches(cfg["batch_size"], generator,
                                  augment=cfg["augment"]):
            for group, base in zip(optimizer.param_groups, base_lrs, strict=True):
                group["lr"] = base * lr_at(step, total_steps, cfg)

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
               "epoch_s": epoch_s, "img_per_s": seen / epoch_s}

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
        history.append(row)

        if run is not None:
            run.log({f"s{seed}/{k}": v for k, v in row.items() if k != "seed"})
        if verbose:
            val_part = f"val={row['val_acc']:.4f}  " if val is not None else ""
            print(f"  [seed {seed}] epoch {epoch:03d}  lr={row['lr']:.4f}  "
                  f"loss={row['train_loss']:.4f}  {val_part}"
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
    }
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

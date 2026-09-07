"""One training run, from a seed to a row of numbers.

Two accuracies get reported and they answer different questions:

  test_acc_final     the last epoch. What you get if you just train and stop.
  test_acc_selected  the epoch with the best validation accuracy. What you get
                     with honest model selection.

A baseline has to state which one it means, so it reports both and never mixes
them. The reference config runs with val_size=0: there is nothing to select on,
the two coincide, and the report says `selection: final_epoch` rather than
pretending a selection happened.

The anchored arms add a third stream of numbers, one row per probe rather than
one per epoch, written to probes.csv/jsonl: lambda, the drift d_t and its target,
||w - w0||, the per-layer weight norms and eta_eff, and lambda_min(H). Those are
logged on EVERY arm, the unanchored baseline included. An anchored run and a
decayed run end at different weight norms and therefore at different effective
learning rates, and without the eta_eff trace on both there is no way after the
fact to tell a real effect from that confound.
"""
import math
import time

import torch
import torch.nn.functional as F

from cifarbase import anchor as anchor_mod
from cifarbase import checkpoint as ckpt_mod
from cifarbase import kernel
from cifarbase.controller import DriftController
from cifarbase.data import class_balanced_indices
from cifarbase.hessian import lanczos_extremes
from cifarbase.metrics import evaluate
from cifarbase.model import build_model, count_params
from cifarbase.utils.seeding import seed_everything

# Which images the probe uses. A constant, never the run seed: every seed and
# every arm has to measure drift on the same batch, or the batch's own difficulty
# ends up inside the across-seed spread of d_t -- the one number the controller
# steers by. Disjoint from the init stream, the batch-order stream and the
# tangent stream, all four of which are separate here on purpose.
PROBE_INDEX_SEED = 20260907


def build_optimizer(model, cfg):
    """SGD, and only SGD.

    Keeping the baseline to one optimiser is the point of the repo: a number you
    can compare against needs a recipe that did not drift. Adding AdamW here is a
    two-line change, and should come with its own config file and its own number.

    The parameter groups are the anchor's roles, and they all carry the SAME
    weight_decay. The split exists so each role can get its own lambda; making it
    also change decay -- the popular "no decay on BatchNorm and biases" recipe --
    would move the baseline this branch is measured against.
    """
    if cfg["optimizer"] != "sgd":
        raise ValueError(f"unknown optimizer {cfg['optimizer']!r}: this baseline "
                         f"is sgd only")
    return torch.optim.SGD(anchor_mod.build_param_groups(model, cfg),
                           lr=cfg["lr"],
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


def train_once(cfg, train, val, test, device, seed, run=None, verbose=True,
               out=None):
    seed_everything(seed, cfg["deterministic"])
    model = build_model(cfg, device, quiet=not verbose)

    # w0 is captured HERE, immediately after init and before the first step, and
    # lives in the model's state_dict as buffers so it is checkpointed and moved
    # with the model. An anchor recreated on a resume would point at wherever the
    # run happened to be interrupted, and every d_t would be measured against
    # that -- both curves would look entirely normal.
    anchor = anchor_mod.Anchor(model, cfg)
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
    probe_every = int(cfg["probe_every"])
    hessian_every = int(cfg["hessian_every"])

    controller = DriftController(cfg, total_steps)
    probe_index = class_balanced_indices(train, int(cfg["probe_batch"]),
                                         PROBE_INDEX_SEED)
    probe_x, probe_y = train.take(probe_index)

    # K_0. Everything the controller does is relative to this, so it is captured
    # once and then only ever read.
    #
    # NOT necessarily at w0, and that is deliberate. With zero_init_residual every
    # residual branch outputs exactly 0 at w0, which leaves about 39% of each
    # block's second ReLU sitting exactly on its kink -- so the NTK is genuinely
    # discontinuous at w0, and one training step of any size moves the sketch by
    # ~24%. A reference taken there is one d_t leaves immediately and can never
    # approach again, which would make the whole d* schedule unreachable. w0
    # itself is still step 0: the anchor and the drift reference are two different
    # objects and only the latter moves. selfcheck's K0 gate measures this.
    reference_step = int(cfg["anchor_reference_step"])
    if reference_step < 0:
        reference_step = int(cfg["anchor_hold_steps"])
    reference = None
    if reference_step == 0:
        reference, _ = kernel.measure(model, probe_x, cfg)

    generator = torch.Generator(device=device).manual_seed(seed)
    best = {"val_acc": -1.0, "epoch": -1, "state": None}
    history, probes = [], []
    start_epoch = 0
    step = 0

    if cfg["resume"]:
        state = ckpt_mod.load(cfg["resume"], model, optimizer, controller,
                              anchor, device, generator, cfg)
        start_epoch, step = state["epoch"] + 1, state["step"]
        reference, probe_index = state["reference"], state["probe_index"]
        probe_x, probe_y = train.take(probe_index)
        history, probes = state["history"], state["probes"]

    if verbose:
        print(f"  {anchor.describe()}")
        print(f"  {controller.describe()}")
        print(f"  probe: {len(probe_index)} fixed images, every {probe_every} "
              f"steps, {cfg['probe_tangents']} tangents, kernel "
              f"{cfg['probe_kernel']}  ({total_steps} steps total)")
        print(f"  K_0 taken at step {reference_step}"
              + ("  (w0 itself; §7 literally)" if reference_step == 0 else
                 "  (end of the hold window: w0 is a ReLU-kink degeneracy, "
                 "see selfcheck's K0 gate)"))

    images = 0
    started = time.time()
    model.train()
    # Carried across epochs, not reset per epoch: probe_every is a step count and
    # can be larger than an epoch, and an epoch row missing the anchor columns
    # would leave a hole in history.csv for no reason.
    latest = {}

    for epoch in range(start_epoch, cfg["epochs"]):
        epoch_start = time.time()
        running = correct = seen = 0
        for x, y in train.batches(cfg["batch_size"], generator,
                                  augment=cfg["augment"]):
            scale = lr_at(step, total_steps, cfg)
            for group, base in zip(optimizer.param_groups, base_lrs, strict=True):
                group["lr"] = base * scale
            lr_now = base_lrs[0] * scale
            lam = controller.lam(step)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y,
                                   label_smoothing=cfg["label_smoothing"])
            loss.backward()
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            # After the step and after the clip, never before either. Clipping
            # first means a large gradient cannot cancel the anchor pull; being
            # outside optimizer.step() means the pull never enters the momentum
            # buffer, so a change in lambda takes effect at once instead of
            # bleeding through the velocity for several more steps.
            if lam > 0.0:
                anchor.pull(anchor.lambdas(lam), lr_now)

            running += float(loss.detach()) * y.numel()
            correct += int((logits.detach().argmax(1) == y).sum())
            seen += y.numel()
            images += y.numel()

            if step % probe_every == 0:
                if reference is None and step >= reference_step:
                    reference, _ = kernel.measure(model, probe_x, cfg)
                    if verbose:
                        print(f"  K_0 captured at step {step}")
                latest = _probe(model, probe_x, probe_y, cfg, anchor, controller,
                                reference, seed, epoch, step, lr_now, lam,
                                hessian_every)
                probes.append(latest)
                if run is not None:
                    run.log({f"s{seed}/probe/{k}": v for k, v in latest.items()
                             if isinstance(v, (int, float))})
            step += 1

        epoch_s = time.time() - epoch_start
        row = {"seed": seed, "epoch": epoch, "step": step,
               "lr": optimizer.param_groups[0]["lr"],
               "train_loss": running / seen, "train_acc_batchwise": correct / seen,
               "epoch_s": epoch_s, "img_per_s": seen / epoch_s}
        # The last probe of the epoch, carried onto the epoch row so the anchor
        # series also land in history.csv and can be plotted against accuracy
        # without joining two files.
        row.update({k: latest[k] for k in _EPOCH_KEYS if k in latest})

        if val is not None:
            val_metrics = evaluate(model, val, eval_bs)
            row["val_loss"] = val_metrics["loss"]
            row["val_acc"] = val_metrics["acc"]
            if val_metrics["acc"] > best["val_acc"]:
                # Kept on CPU: 11M params is 45 MB, and a GPU copy on every
                # improvement is needless pressure on a card already holding the
                # whole dataset. w0 is stripped because it is a second full copy
                # of the parameters and is already in the checkpoint.
                best = {"val_acc": val_metrics["acc"], "epoch": epoch,
                        "state": {k: v.detach().to("cpu", copy=True) for k, v
                                  in anchor_mod.strip_w0(model.state_dict()).items()}}

        test_metrics = evaluate(model, test, eval_bs)
        row["test_loss"] = test_metrics["loss"]
        row["test_acc"] = test_metrics["acc"]
        history.append(row)

        if run is not None:
            run.log({f"s{seed}/{k}": v for k, v in row.items()
                     if k != "seed" and isinstance(v, (int, float))})
        if verbose:
            val_part = f"val={row['val_acc']:.4f}  " if val is not None else ""
            anchor_part = (f"lam={latest['lam']:.3g} d={latest['d']:.4f}"
                           f"/{latest['d_target']:.4f}  "
                           if "d" in latest else "")
            print(f"  [seed {seed}] epoch {epoch:03d}  lr={row['lr']:.4f}  "
                  f"loss={row['train_loss']:.4f}  {val_part}{anchor_part}"
                  f"test={row['test_acc']:.4f}  {row['epoch_s']:.1f}s")

        stopping = (cfg["stop_after_epoch"]
                    and epoch + 1 >= int(cfg["stop_after_epoch"]))
        if out and cfg["ckpt_every"] and (
                stopping or (epoch + 1) % int(cfg["ckpt_every"]) == 0):
            ckpt_mod.save(f"{out}/ckpt_s{seed}.pt", cfg=cfg, seed=seed,
                          epoch=epoch, step=step, model=model,
                          optimizer=optimizer, controller=controller,
                          reference=reference, probe_index=probe_index,
                          history=history, probes=probes, generator=generator)
        if stopping:
            # Deliberately after the checkpoint, and deliberately not a change to
            # `epochs`: the lr schedule still belongs to the full run, so the
            # continuation picks up the same curve at the same point.
            if verbose:
                print(f"  [seed {seed}] stopping after epoch {epoch} of "
                      f"{cfg['epochs']} as asked; resume with "
                      f"--resume {out}/ckpt_s{seed}.pt")
            break

    wall = time.time() - started
    final = _describe(model, cfg, train, test)

    # Re-measure on the selected checkpoint rather than reusing the epoch row:
    # the row only has test loss and accuracy, and reporting a selected accuracy
    # next to final-epoch per-class numbers would be mixing two models.
    if best["state"] is not None:
        live = {k: v.detach().clone()
                for k, v in anchor_mod.strip_w0(model.state_dict()).items()}
        # strict=False: w0 was stripped from both snapshots and stays as it is,
        # which is correct -- w0 is a constant of the run, not part of the state
        # being selected over.
        model.load_state_dict(best["state"], strict=False)
        selected = _describe(model, cfg, train, test)
        model.load_state_dict(live, strict=False)
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
    summary.update(controller.summary())
    if probes:
        # .get, because a run shorter than anchor_reference_step never captures
        # K_0 and so has no drift columns at all -- which is a legitimate state
        # for a two-epoch smoke test, not something to crash on.
        summary.update({f"{k}_final": probes[-1][k]
                        for k in ("d", "d_ntk", "d_feature", "a_ntk",
                                  "a_feature", "scale_ntk", "scale_feature",
                                  "dist", "dist_rel") if k in probes[-1]})
    # The names behind the w_norms / eta_eff vectors, recorded once per run: the
    # vectors are useless six months later without them.
    summary["anchor_layers"] = anchor.layer_names

    if verbose:
        print(f"  [seed {seed}] {wall:.0f}s  "
              f"test_final={summary['test_acc_final']:.4f}  "
              f"test_selected={summary['test_acc_selected']:.4f}  "
              f"train={summary['train_acc']:.4f}  "
              f"gap={summary['gen_gap']:+.4f}")
    if run is not None:
        run.log({f"summary/s{seed}/{k}": v for k, v in summary.items()
                 if isinstance(v, (int, float))})

    if out and cfg["ckpt_every"] and not cfg["stop_after_epoch"]:
        ckpt_mod.save(f"{out}/ckpt_s{seed}.pt", cfg=cfg, seed=seed,
                      epoch=cfg["epochs"] - 1, step=step, model=model,
                      optimizer=optimizer, controller=controller,
                      reference=reference, probe_index=probe_index,
                      history=history, probes=probes, generator=generator)
    return summary, history, probes


# Which probe columns ride along on the epoch row. Deliberately the scalars only:
# the per-layer vectors belong in probes.csv, where they do not widen every
# history row by sixty columns.
_EPOCH_KEYS = ("lam", "d", "d_ema", "d_target", "d_ntk", "d_feature", "a_ntk",
               "scale_ntk", "dist", "dist_rel", "eta_eff_mean",
               "hessian_lambda_min")


def _probe(model, probe_x, probe_y, cfg, anchor, controller, reference, seed,
           epoch, step, lr_now, lam_applied, hessian_every):
    """One row of the anchor stream. Read-only with respect to the model.

    Nothing in here may touch a global RNG: with anchor_mode off this whole path
    has to be inert, and gate H2 checks bit-for-bit that it is. The tangents come
    from their own generator, the probe indices from theirs, and the Lanczos
    start vector is a constant.
    """
    _, drift = kernel.measure(model, probe_x, cfg, reference=reference)
    row = {"seed": seed, "epoch": epoch, "step": step,
           "progress": step / max(controller.total_steps, 1),
           "lr": lr_now, "lam_applied": lam_applied}
    row.update(drift)
    # Before K_0 exists there is no drift to report, and feeding the controller a
    # zero would look to it like a run that is not learning at all. The rows are
    # still written -- lambda, the norms and eta_eff are all meaningful there --
    # with the drift columns simply absent.
    if not drift:
        row["lam"] = controller.lam(step)
        row.update(anchor.distance(lr_now))
        return row
    row["d"] = kernel.selected(drift, cfg["probe_kernel"], cfg["probe_signal"])
    row.update(controller.observe(step, row["d"]))
    row.update(anchor.distance(lr_now))
    if hessian_every and step % hessian_every == 0:
        row.update(lanczos_extremes(model, probe_x, probe_y,
                                    int(cfg["hessian_iters"]),
                                    bool(cfg["probe_deterministic"])))
    return row


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

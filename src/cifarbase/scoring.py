"""How hard each training example is, and when the curriculum lets it in.

Two things live here, deliberately separated because they change on different
timescales:

  * the ORDER, computed once per run -- it never changes;
  * lambda(t), the share of that order visible at epoch t -- how far the
    curriculum has opened up, recomputed every epoch and costing one slice.

The difficulty of an example is one number:

    margin = logit of the true class - the best competing logit

One quantity, read two ways, which is the whole reason not to compute two: a
large positive margin means easy, and `margin < 0` means the scorer gets the
example wrong. Sorting by margin gives the curriculum; the sign of it gives the
misclassified set. Measured on the UN-augmented image with the model in eval(),
like every other number this repo reports about a model.

WHEN to score matters as much as how. After 100 epochs a ResNet-18 is right
about 99.9% of its own training set, so `margin < 0` would pick out a few dozen
images out of 50 000 and the easy/hard split would stop meaning anything.
`score_epoch` takes the reading early instead, while the model knows something
but has not yet memorised the set -- which under label noise is also before it
has memorised the wrong labels, and that is what keeps the easy end of the
ordering the correctly labelled end.

The scorer was trained on the examples it scores, so its margins leak a little
memorisation. Early in training that leak is small, and it is a stated
limitation rather than something this file pretends to solve -- cross-fitting
the scores is the fix, and it is a separate experiment.

THE PATH. lambda(t) goes from `easy_frac` at epoch 0 to 1.0 at `ramp_epochs`,
and stays at 1.0 after that. Its SHAPE is `pacing`: `linear` walks it in equal
increments, `quadratic` and `exp` linger on the easy end, `root` opens fast then
slows, and `step` holds at easy_frac and jumps -- which is exactly the two-stage
path of Bengio et al. (ICML 2009, section 5). Walking the path in increments is
what a continuation method actually does (descend a little, deform a little,
descend a little); their jump is one value of this knob rather than a different
experiment.

WHAT THE PACING COSTS. An epoch is always a full split's worth of steps, so a
pool of size k*N is drawn 1/k times per example per epoch: at lambda 0.5 every
visible example is seen twice while the baseline sees each of its once. That
repetition is not a bug and not an artefact of this repo -- it is what a pacing
function DOES under a fixed step budget, and it is the protocol Wu, Dyer &
Neyshabur (ICLR 2021) sweep. But it means a curriculum arm differs from the
baseline in two ways at once, and only `random_order` -- the same pacing with a
shuffled scoring -- separates the two. `exposure_profile` puts numbers on it.
"""
import glob
import math
import os

import torch

from cifarbase.data import corrupted_mask

# The four arms. `easy_first` and `hard_first` are the same scoring read in
# opposite directions; `random_order` keeps the pacing and throws the scoring
# away, which is what makes it the control rather than a curiosity.
MODES = ("none", "easy_first", "hard_first", "random_order")

# lambda(t) for t = epoch/ramp_epochs in [0, 1) and b = easy_frac. Each is b at
# t=0 and 1.0 at t=1; what differs is the shape between. `step` stays at b and
# lets lambda_at's t>=1 branch do the jump -- Bengio et al.'s two-stage path.
PACINGS = {
    "linear": lambda t, b: b + (1.0 - b) * t,
    "quadratic": lambda t, b: b + (1.0 - b) * t * t,
    "root": lambda t, b: b + (1.0 - b) * math.sqrt(t),
    "exp": lambda t, b: b ** (1.0 - t),
    "step": lambda t, b: b,
}

# Orderings must not move with the run seed: the curriculum is a fixed property
# of the experiment, the way a scores file is. random_order is the exception --
# a control that always drew the same permutation would measure one particular
# ordering rather than the absence of one -- so it takes the run seed, offset
# far from it so the two generators cannot align.
_ORDER_SEED = 20210131
_TEACHER_SEED = 20210132


# --------------------------------------------------------------------------
# the path
# --------------------------------------------------------------------------

def lambda_at(cfg, epoch):
    """The share of the sorted train set visible at `epoch`, in (0, 1].

    A `ramp_epochs` of 0 leaves lambda at 1 from the first epoch, which IS the
    baseline -- that is what makes the no-curriculum arm a value of these knobs
    rather than a separate recipe.
    """
    r = cfg["ramp_epochs"]
    if r <= 0:
        return 1.0
    t = epoch / r
    if t >= 1.0:
        return 1.0
    lo = float(cfg["easy_frac"])
    return min(1.0, PACINGS[cfg.get("pacing", "linear")](t, lo))


# --------------------------------------------------------------------------
# the scores
# --------------------------------------------------------------------------

@torch.no_grad()
def margins(model, split, batch_size=1000):
    """One float per example, in the split's own order.

    Leaves the model in the mode it arrived in, for the same reason `evaluate`
    does: scoring mid-run must not quietly hand BatchNorm a batch of running
    statistics it was never supposed to see.
    """
    was_training = model.training
    model.eval()

    out = []
    for x, y in split.chunks(batch_size):
        logits = model(x)
        true = logits.gather(1, y[:, None]).squeeze(1)
        # scatter, not scatter_: masking the true logit must not damage the
        # tensor we just read it out of.
        others = logits.scatter(1, y[:, None], float("-inf"))
        out.append((true - others.max(1).values).cpu())

    if was_training:
        model.train()
    return torch.cat(out)


def save_scores(scores, path, cfg, seed, epoch):
    """The margins, plus the run that produced them.

    A bare tensor is ambiguous in the ways that matter: a scores file from a
    clean run and one from a 20%-noise run are both 50 000 floats, and
    `load_scores` could only ever check the count. Recording the budget and the
    noise turns a silent confound into a printed line.
    """
    torch.save({"margins": scores, "epoch": epoch, "seed": seed,
                "epochs": cfg["epochs"], "config": cfg["config"],
                "label_noise": cfg["label_noise"]}, path)


def _mounted():
    """Every scores_s*.pt under the Kaggle mounts, sorted."""
    return sorted(glob.glob("/kaggle/input/**/scores_s*.pt", recursive=True))


def has_scores(path):
    """Is there a scores file to load? Asked by _validate, which runs long
    before the split exists and so cannot check anything but availability."""
    return bool(path and os.path.isfile(path)) or len(_mounted()) == 1


def _resolve(path):
    """A scores path, or the single scores_s*.pt under the Kaggle mounts.

    Mirrors `_find_root` in data.py, and for the same reason: the mount point is
    not known until the dataset is attached, so naming it in a YAML costs a whole
    push cycle. Refusing when several match is the point -- silently picking one
    of two is exactly the failure this file exists to prevent.
    """
    if path and os.path.isfile(path):
        return path
    hits = _mounted()
    if len(hits) == 1:
        print(f"scores: {hits[0]} (resolved from the Kaggle mounts)")
        return hits[0]
    if len(hits) > 1:
        raise SystemExit(f"!! {len(hits)} scores files under /kaggle/input; name "
                         f"one explicitly with --scores: {', '.join(hits)}")
    return path


def load_scores(path, expected):
    """The margin vector written by a scoring run, checked against the split.

    The check is the important part. Scores index into the train split by
    position, so they stay valid across runs only while the split is built the
    same way -- change `train_subset`, `val_size` or `label_noise` between the
    scoring run and the curriculum run and every index silently points at a
    different image, or at the same image with a different label.

    Accepts both the dict `save_scores` writes and a bare tensor, so files
    produced before that format existed still load.
    """
    path = _resolve(path)
    if not path:
        raise SystemExit("!! no scores file given: run once with --dump-scores "
                         "and pass the resulting .pt with --scores")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        scores = blob["margins"]
        print(f"  from epoch {blob['epoch']} of a {blob['epochs']}-epoch "
              f"{blob.get('config', '?')!r} run, seed {blob['seed']}, "
              f"label_noise {blob.get('label_noise', '?')}")
    else:
        scores = blob
        print("  bare tensor: no provenance recorded, check it yourself")
    if scores.numel() != expected:
        raise SystemExit(f"!! {path} holds {scores.numel()} scores but the train "
                         f"split has {expected}: train_subset or val_size "
                         f"changed since the scoring run")
    return scores


# --------------------------------------------------------------------------
# the order, and the pool it yields per epoch
# --------------------------------------------------------------------------

def _k_at(n, lam):
    return max(1, min(n, round(lam * n)))


def curriculum_order(cfg, train, device, seed=0, scores=None, verbose=True):
    """The FULL train index set, sorted so index 0 enters first -- or None.

    Sorted once because the ordering never changes: the per-epoch pool is a
    prefix of this, which costs a slice rather than a 50k argsort per epoch.

    None rather than "every index" is deliberate and load-bearing: `Split.batches`
    then takes exactly the path it took before this file existed, so
    `curriculum: none` is not merely equivalent to the baseline, it is the
    baseline, down to the draws it makes from the generator.

    `scores` is the teacher's margins when a study fitted them in process;
    without it they come from the file named by cfg["scores"]. Either way the
    ordering is the same for every seed, because it is a property of the
    experiment and not of the run -- except under random_order, where varying it
    with the seed is the whole point.
    """
    if (cfg["curriculum"] == "none" or cfg["ramp_epochs"] <= 0
            or cfg["easy_frac"] >= 1.0):
        return None

    n = len(train)
    mode = cfg["curriculum"]
    if mode == "random_order":
        # No difficulty information at all, and the same pacing as the arms it
        # controls. Whatever this arm gains over the baseline was bought by the
        # pacing -- the repetition and the smaller pool -- and not by knowing
        # which examples are easy.
        gen = torch.Generator().manual_seed(_ORDER_SEED + int(seed))
        order = torch.randperm(n, generator=gen)
        scores = None
    else:
        if scores is None:
            scores = load_scores(cfg["scores"], n)
        scores = scores.detach().to("cpu").float()
        if scores.numel() != n:
            raise SystemExit(f"!! {scores.numel()} scores for a train split of "
                             f"{n}: the split changed since they were made")
        # Descending margin is easiest first; hard_first is the same ranking
        # read from the other end, which is what makes "the direction of the
        # order" a testable thing rather than an opinion.
        order = torch.argsort(scores, descending=(mode == "easy_first"))

    if verbose:
        _describe_order(cfg, order, scores, n)

    return order.to(device)


def _describe_order(cfg, order, scores, n):
    """What the arm is about to do, in band, before it does it."""
    lam0 = lambda_at(cfg, 0)
    k = _k_at(n, lam0)
    pool = order[:k]
    print(f"curriculum: {cfg['curriculum']}, {cfg['pacing']} pacing, "
          f"lambda {lam0:.2f} -> 1.00 over {cfg['ramp_epochs']} epochs "
          f"({k}/{n} examples at epoch 0)")

    seen = exposure_profile(cfg, n)
    print(f"  fixed step budget: epoch 0 draws each visible example "
          f"{n / k:.2f}x. Over the run the easy decile of this order gets "
          f"{seen['easy_decile']:.1f} exposures and the hard decile "
          f"{seen['hard_decile']:.1f}, against {seen['baseline']:.1f} each for "
          f"the baseline")

    if scores is not None:
        print(f"  the scorer gets {float((scores[pool] < 0).float().mean()):.1%} "
              f"of that initial pool wrong")
    noise = float(cfg["label_noise"])
    if noise > 0:
        # The claim the whole noisy-label design rests on: under noise the easy
        # end of the ordering is the correctly labelled end. Printed rather than
        # assumed. If this is not well below `noise`, the scoring separated
        # nothing and no number downstream means what it says it does. Under
        # random_order it should land ON `noise` -- the control working.
        bad = corrupted_mask(n, noise)
        print(f"  {float(bad[pool].float().mean()):.1%} of it carries a "
              f"corrupted label ({noise:.1%} of the whole train set does)")


def exposure_profile(cfg, n):
    """How often each rank of the ordering is drawn, over the whole run.

    An epoch is `n` draws whatever the pool holds, so an example inside a pool
    of k is drawn n/k times that epoch and one outside it zero times. Summed
    over the run, this is the repetition a fixed step budget hands the easy end
    of the order: the SECOND way a curriculum arm differs from the baseline, and
    the reason random_order has to exist.

    Approximate to one epoch of draws per epoch (the last partial batch is
    dropped), and a decile is represented by its midpoint rank, which is fair
    because the count is monotone in the rank.
    """
    epochs = int(cfg["epochs"])
    schedule = [_k_at(n, lambda_at(cfg, e)) for e in range(epochs)]
    rates = [(k, n / k) for k in schedule]

    def at(rank):
        return sum(rate for k, rate in rates if rank < k)

    return {"baseline": float(epochs),
            "easy_decile": at(int(0.05 * n)),
            "median": at(n // 2),
            "hard_decile": at(int(0.95 * n)),
            "pool_k": schedule}


def fit_teacher(cfg, train, test, device, verbose=True):
    """Cross-fit difficulty scores: nothing is ranked by a model that saw it.

    `teacher_folds` models, each trained on the complement of the fold it
    scores, their margins written back into one vector of length n.

    This is what lets the ranking be read off a CONVERGED model. The self-scored
    path cannot: a network is right about ~99.9% of its own train set by the
    end, and under label noise it has memorised the wrong labels too, so the
    reading has to happen early, `score_epoch` becomes a delicate knob, and a
    leak survives in the ranking anyway. A held-out model has neither problem --
    and under noise it is strictly better at spotting a corrupted label, because
    a wrong label it never trained on simply disagrees with the rest of the data.

    The teacher trains on the same corrupted labels the arms do. That is the
    honest setting: a curriculum that needed clean labels to build its ordering
    would be assuming away the problem it is supposed to help with.
    """
    from cifarbase.train import train_once

    n = len(train)
    folds = int(cfg["teacher_folds"])
    tcfg = dict(cfg, epochs=int(cfg["teacher_epochs"]), curriculum="none",
                ramp_epochs=0, dump_scores=False, wandb=False, per_class=False,
                scores="", arm="teacher")
    # The teacher's schedule has to finish inside the teacher's own budget:
    # borrowing a 30-epoch recipe's warmup for an 8-epoch teacher would leave
    # its lr high at the end and make the margins noisier than they need to be.
    tcfg["warmup_epochs"] = min(int(cfg["warmup_epochs"]),
                                max(0, tcfg["epochs"] // 4))

    gen = torch.Generator().manual_seed(_TEACHER_SEED)
    perm = torch.randperm(n, generator=gen)
    scores = torch.zeros(n, dtype=torch.float32)
    accs, scales = [], []

    if verbose:
        print(f"teacher: {folds}-fold cross-fit, {tcfg['epochs']} epochs each, "
              f"{tcfg['arch']} on {n} examples, "
              f"label_noise {float(cfg['label_noise']):.0%}")
    for fold in range(folds):
        held = perm[fold::folds]
        keep = torch.ones(n, dtype=torch.bool)
        keep[held] = False
        keep = keep.nonzero(as_tuple=True)[0]

        fit = train.subset(keep.to(device))
        out = train.subset(held.to(device))
        summary, _, model = train_once(tcfg, fit, None, test, device,
                                       _TEACHER_SEED + fold, verbose=False,
                                       return_model=True)
        fold_scores = margins(model, out, cfg["eval_batch_size"]).float()
        scores[held] = fold_scores
        accs.append(summary["test_acc_final"])
        # One ranking is built out of several models' margins, so the folds have
        # to be on the same scale for a cross-fold comparison to mean anything.
        # Not rescaled -- the sign of a margin is load-bearing downstream, and
        # standardising it away would cost more than it buys -- but reported, so
        # folds that disagree about the scale are visible rather than silently
        # ranked against each other.
        scales.append((float(fold_scores.mean()), float(fold_scores.std())))
        if verbose:
            print(f"  fold {fold}: trained on {len(fit)}, scored {len(out)}, "
                  f"test acc {summary['test_acc_final']:.4f}, "
                  f"margin {scales[-1][0]:+.2f} +- {scales[-1][1]:.2f}")
        del fit, out, model

    info = {"folds": folds, "epochs": tcfg["epochs"], "test_acc": accs,
            "margin_mean_std": scales,
            "misclassified": float((scores < 0).float().mean()),
            "label_noise": float(cfg["label_noise"])}
    if verbose:
        spread = max(m for m, _ in scales) - min(m for m, _ in scales)
        print(f"  the teacher misclassifies {info['misclassified']:.1%} of the "
              f"train set: that is the hard end of the order")
        print(f"  fold margin means differ by {spread:.2f}; large next to the "
              f"within-fold spread would mean the folds are being ranked "
              f"against each other on different scales")
    return scores, info


def pool_at(order, epoch, cfg):
    """The prefix visible at `epoch`, or None once the whole set is visible.

    Returning None at lambda >= 1 is not an optimisation. A full-length pool
    would still go through `_epoch_order`, whose `pool[randperm(N)]` is a
    permutation of the SORTED order and therefore a different sequence of images
    from the baseline's `randperm(N)` -- the same generator draws, a different
    epoch. Past the end of the ramp the run has to BE the baseline, exactly, or
    the two arms stop coinciding at the one place they are supposed to.
    """
    if order is None:
        return None
    lam = lambda_at(cfg, epoch)
    if lam >= 1.0:
        return None
    return order[:_k_at(order.numel(), lam)]

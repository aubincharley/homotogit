"""Training objectives, as a named family the configuration can select.

The benchmark trained every method with plain softmax cross-entropy.  Swapping
that out asks a question the four frozen methods cannot answer on their own:
**is the curriculum gain a property of the intervention, or of the objective it
was measured under?**  If +4.5 pp survives a loss whose gradient has a different
shape, the effect is about the operator; if it does not, it is about the pair.

Three constraints shaped what is here.

**Every loss must be a per-sample mean.**  ``Trainer._step`` accumulates the
gradient from microbatches of 32 and rescales each by ``n_micro / n_batch``, so
the sum equals the full-batch gradient *only* for objectives that average over
samples independently.  Anything with a batch-level coupling (contrastive
objectives, losses normalised by batch statistics) would silently compute a
different gradient at a different microbatch size.  ``build`` therefore refuses
names it does not implement rather than accepting a callable.

**Cross-entropy stays reported whatever is optimised.**  ``evaluate`` returns
``ce`` for every run and ``obj`` alongside it, so a column trained on the square
loss is still comparable to the cross-entropy column on a common scale.  Test
accuracy is objective-agnostic and remains the primary comparator.

**The square loss is taken on the logits, not on the softmax.**  Those are two
different objectives and only the first is the one the literature result is
about (Hui & Belkin, ICLR 2021).  On probabilities it is the Brier score, whose
gradient passes through the softmax Jacobian and therefore *saturates*: a
confidently wrong prediction receives almost no gradient.  On logits the
gradient is ``2(z - y)/C``, linear in the error and never saturating, which is
the whole reason to try it.  ``square`` is the logit form; the Brier score is
available as ``brier`` for the contrast, and is not recommended as a default.

One consequence to carry into the run plan: the square loss does not share
cross-entropy's gradient scale.  Cross-entropy's gradient with respect to the
logits is ``(p - y)``, bounded in ``[-1, 1]``; the square loss's is unbounded in
``z``.  The reference learning rate is therefore not automatically the right one
for it, and a poor result at a fixed learning rate would not distinguish the
objective from the step size.  Sweep the learning rate for ``square``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

#: every name ``build`` accepts
LOSSES = ("cross_entropy", "label_smoothing", "square", "focal", "brier")


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)


def label_smoothing(logits: torch.Tensor, target: torch.Tensor, *,
                    epsilon: float = 0.1) -> torch.Tensor:
    return F.cross_entropy(logits, target, label_smoothing=float(epsilon))


def square(logits: torch.Tensor, target: torch.Tensor, *,
           normalise_by_classes: bool = True) -> torch.Tensor:
    """``|| z - onehot(y) ||^2``, optionally divided by the number of classes.

    The division is a convention, not a detail: it moves the effective learning
    rate by a factor of ``C``.  It is recorded in the configuration so a run can
    be reproduced, and it is on by default because that is the form in which the
    square-loss result is usually stated.
    """
    t = F.one_hot(target, logits.shape[1]).to(logits.dtype)
    per_sample = ((logits - t) ** 2).sum(dim=1)
    if normalise_by_classes:
        per_sample = per_sample / logits.shape[1]
    return per_sample.mean()


def focal(logits: torch.Tensor, target: torch.Tensor, *,
          gamma: float = 2.0) -> torch.Tensor:
    """``-(1 - p_t)^gamma log p_t``.  At ``gamma = 0`` this is cross-entropy."""
    logp_t = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
    return (-((1.0 - logp_t.exp()) ** float(gamma)) * logp_t).mean()


def brier(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Square loss on the softmax.  Saturates; kept for contrast with ``square``."""
    t = F.one_hot(target, logits.shape[1]).to(logits.dtype)
    return ((logits.softmax(dim=1) - t) ** 2).sum(dim=1).mean()


def build(cfg) -> callable:
    """The objective named by ``cfg`` as ``f(logits, target) -> scalar``.

    ``cfg`` is a ``LossConfig``; its unused fields are ignored, so one config
    shape serves every objective and the run record always states all of them.
    """
    name = cfg.name
    if name not in LOSSES:
        raise ValueError("unknown loss %r; known: %s" % (name, ", ".join(LOSSES)))
    if name == "cross_entropy":
        return cross_entropy
    if name == "label_smoothing":
        eps = float(cfg.label_smoothing)
        if not 0.0 <= eps < 1.0:
            raise ValueError("label_smoothing must be in [0, 1), got %r" % (eps,))
        return lambda z, y: label_smoothing(z, y, epsilon=eps)
    if name == "square":
        norm = bool(cfg.normalise_by_classes)
        return lambda z, y: square(z, y, normalise_by_classes=norm)
    if name == "focal":
        g = float(cfg.gamma)
        if g < 0.0:
            raise ValueError("focal gamma must be >= 0, got %r" % (g,))
        return lambda z, y: focal(z, y, gamma=g)
    return brier

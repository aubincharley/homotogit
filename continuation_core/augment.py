"""Input augmentation, pinned the way everything else in this benchmark is.

The recorded runs use **no augmentation** -- ``data.py`` says so, and the control
reaches 75.9 % where a ResNet-20 on CIFAR-10 with the standard recipe reaches
about 91-92 %.  That gap is the reason this module exists: random crop and
horizontal flip are themselves *input-space* interventions, like the blur and the
resolution reduction, so whether the curriculum gain survives them is a question
about the result, not about tooling.

The recipe is the standard one (He et al. 2016, sec. 4.2): pad four pixels of
zeros on each side, take a random 32x32 crop, flip horizontally with probability
one half.  Test images are never touched -- ``evaluate`` calls the pipeline
directly and never comes through here, so the train probe measures clean images
too, which is what makes a train/test gap readable.

**Reproducibility is the whole design constraint.**  Everything else in a run is
pinned: the training subset, the probe indices, the per-epoch permutations, the
initial weights.  Augmentation adds one random draw per image per epoch, and if
it came from the ambient RNG a run would stop being reproducible and, worse,
would not survive a checkpoint resume -- the global RNG has been advanced an
unknown number of times by then.  Every draw here is therefore a pure function of
``(run seed, epoch, batch index, microbatch offset)`` via ``derive_seed``, on a
CPU generator so the values do not depend on the device.  Resuming mid-epoch
reproduces the same crops.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .seeding import derive_seed

#: every recipe ``build`` accepts
AUGMENTATIONS = ("none", "crop_flip")


def _draws(n: int, pad: int, flip_p: float, seed: int):
    """Offsets and flip mask for ``n`` images, on the CPU, from a pinned seed."""
    g = torch.Generator().manual_seed(int(seed))
    span = 2 * pad + 1
    dy = torch.randint(span, (n,), generator=g)
    dx = torch.randint(span, (n,), generator=g)
    flip = torch.rand(n, generator=g) < flip_p
    return dy, dx, flip


def crop_flip(x: torch.Tensor, *, pad: int, flip_p: float, seed: int) -> torch.Tensor:
    """Pad with zeros, take a random same-size crop, flip half the images.

    ``x`` is ``[N, C, H, W]`` uint8 and the result is uint8 of the same shape, so
    this sits before the normalisation pipeline and leaves it untouched.
    """
    if x.ndim != 4:
        raise ValueError("expected [N, C, H, W], got %r" % (tuple(x.shape),))
    n, c, h, w = x.shape
    dev = x.device
    dy, dx, flip = _draws(n, pad, flip_p, seed)
    dy, dx, flip = dy.to(dev), dx.to(dev), flip.to(dev)

    padded = F.pad(x, (pad, pad, pad, pad))
    rows = dy[:, None] + torch.arange(h, device=dev)[None, :]       # [N, H]
    cols = dx[:, None] + torch.arange(w, device=dev)[None, :]       # [N, W]
    out = padded[torch.arange(n, device=dev)[:, None, None, None],
                 torch.arange(c, device=dev)[None, :, None, None],
                 rows[:, None, :, None],
                 cols[:, None, None, :]]
    return torch.where(flip[:, None, None, None], out.flip(-1), out)


def build(cfg):
    """The augmentation named by ``cfg`` as ``f(images, seed) -> images``.

    ``cfg`` is a ``DataConfig``.  ``"none"`` returns ``None`` rather than an
    identity function, so the training loop can skip the call entirely and the
    recorded runs stay bit-for-bit what they were.
    """
    name = cfg.augmentation
    if name not in AUGMENTATIONS:
        raise ValueError("unknown augmentation %r; known: %s"
                         % (name, ", ".join(AUGMENTATIONS)))
    if name == "none":
        return None
    pad, p = int(cfg.augment_pad), float(cfg.augment_flip_p)
    if pad < 0:
        raise ValueError("augment_pad must be >= 0, got %r" % (pad,))
    if not 0.0 <= p <= 1.0:
        raise ValueError("augment_flip_p must be in [0, 1], got %r" % (p,))
    return lambda images, seed: crop_flip(images, pad=pad, flip_p=p, seed=seed)


def batch_seed(run_seed: int, epoch: int, batch_index: int, offset: int) -> int:
    """A draw identified by where it happens, never by how many draws preceded it."""
    return derive_seed(int(run_seed),
                       "augment::e%d::b%d::m%d" % (int(epoch), int(batch_index), int(offset)))

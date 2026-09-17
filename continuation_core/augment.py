"""Training-image augmentation: zero padding 4, uniform 32x32 crop, horizontal flip (p = 1/2).

He et al. (2016, Sec. 4.2) CIFAR recipe, applied to uint8 images **before** the input
normalisation, on the device, for training batches only (``evaluate`` never calls it).

Draws
-----
For run seed ``s`` and zero-based epoch ``e``, one ``numpy.random.Generator(PCG64)`` seeded
with ``derive_seed(s, stream % e)`` draws, in this order, ``dx`` then ``dy`` (integers in
``0..2*pad``, shape ``[N]``) and ``flip`` (integers in ``{0, 1}``, shape ``[N]``).  Entry ``i``
belongs to example ``i`` of the training tensor, whatever batch it lands in.  The draws are
therefore a pure function of ``(seed, epoch, example)``: every arm of a seed sees the same crop
and flip of the same image in the same epoch, independent of batch size, microbatching, the
intervention, SDPoint's own counter-based stream, and every global torch/numpy/python RNG.
Resumption needs no extra state.

Crop offset ``(dx, dy)`` selects columns ``dx..dx+31`` and rows ``dy..dy+31`` of the padded
40x40 image; the flip reverses the width axis of the cropped image.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .seeding import derive_seed

KINDS = ("none", "pad4_crop32_hflip")
PAD = 4


def crop_flip_draws(seed: int, epoch: int, n: int, stream: str) -> dict:
    g = np.random.Generator(np.random.PCG64(derive_seed(int(seed), stream % int(epoch))))
    dx = g.integers(0, 2 * PAD + 1, size=int(n), dtype=np.int64)
    dy = g.integers(0, 2 * PAD + 1, size=int(n), dtype=np.int64)
    flip = g.integers(0, 2, size=int(n), dtype=np.int64)
    return {"dx": dx, "dy": dy, "flip": flip}


def apply_crop_flip(images_uint8: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor,
                    flip: torch.Tensor) -> torch.Tensor:
    """``images_uint8`` [B, C, H, W]; ``dx``, ``dy``, ``flip`` [B] int64 on the same device."""
    b, c, h, w = images_uint8.shape
    padded = F.pad(images_uint8, (PAD, PAD, PAD, PAD), mode="constant", value=0)
    ar_h = torch.arange(h, device=images_uint8.device)
    ar_w = torch.arange(w, device=images_uint8.device)
    rows = (dy.view(b, 1) + ar_h.view(1, h)).view(b, 1, h, 1).expand(b, c, h, w + 2 * PAD)
    cols = dx.view(b, 1) + ar_w.view(1, w)
    cols = torch.where(flip.view(b, 1).bool(), cols.flip(1), cols).view(b, 1, 1, w).expand(b, c, h, w)
    return padded.gather(2, rows).gather(3, cols)


class EpochAugmenter:
    """Caches one epoch's draws on the device and augments batches of example indices."""

    def __init__(self, kind: str, seed: int, n: int, device, stream: str):
        if kind not in KINDS:
            raise KeyError("unknown augmentation %r (available: %s)" % (kind, KINDS))
        self.kind, self.seed, self.n, self.device, self.stream = kind, int(seed), int(n), device, stream
        self._epoch, self._draws = None, None

    @property
    def active(self) -> bool:
        return self.kind != "none"

    def draws(self, epoch: int) -> dict:
        if self._epoch != int(epoch):
            d = crop_flip_draws(self.seed, epoch, self.n, self.stream)
            self._draws = {k: torch.as_tensor(v, device=self.device) for k, v in d.items()}
            self._epoch = int(epoch)
        return self._draws

    def __call__(self, images_uint8: torch.Tensor, idx: torch.Tensor, epoch: int) -> torch.Tensor:
        if not self.active:
            return images_uint8
        d = self.draws(epoch)
        return apply_crop_flip(images_uint8, d["dx"][idx], d["dy"][idx], d["flip"][idx])

    def describe(self) -> dict:
        return {"kind": self.kind, "pad": PAD, "padding_value": 0, "crop": "uniform 32x32 of the 40x40 padded image",
                "flip": "horizontal, probability 1/2", "stream": self.stream,
                "keyed_by": "(run seed, zero-based epoch); entry i is example i of the training tensor",
                "applied": "uint8, before normalisation, training batches only"}

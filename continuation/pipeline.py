"""Input pipeline: uint8 -> float[0,1] -> ``T_eta`` -> channel normalization.

Ordering is fixed and identical in every run:

1. ``uint8 / 255`` into float ``[0,1]`` (never back to 8-bit afterwards);
2. the transformation ``T_eta``, applied to the **original** image;
3. channel normalization with statistics fitted once on the *unfiltered*
   training subset and shared by every condition.

Because step 3 uses the same constants at every transformation level, a change
in the input statistics caused by smoothing is a property of the objective under
study, not an artefact of a per-condition normalization.
"""
from __future__ import annotations

import time

import torch


class ChannelNormalizer:
    """``(x - mean) / std`` per channel, with fixed constants."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)

    def to(self, device) -> "ChannelNormalizer":
        return ChannelNormalizer(self.mean.flatten().to(device), self.std.flatten().to(device))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)

    def describe(self) -> dict:
        return {"mean": [round(float(v), 8) for v in self.mean.flatten()],
                "std": [round(float(v), 8) for v in self.std.flatten()],
                "fitted_on": "training subset only, unfiltered images",
                "applied": "after the transformation"}


class InputPipeline:
    """Turns a uint8 image batch into model input at a given native parameter."""

    def __init__(self, transform, normalizer: ChannelNormalizer, sync_timing: bool = False):
        self.transform = transform
        self.normalizer = normalizer
        self.sync_timing = sync_timing
        self.transform_seconds = 0.0     # cumulative transformation/preprocessing cost

    @staticmethod
    def to_unit_float(images_uint8: torch.Tensor) -> torch.Tensor:
        return images_uint8.to(torch.float32) / 255.0

    def _sync(self, device):
        if self.sync_timing and device.type == "cuda":
            torch.cuda.synchronize(device)

    def __call__(self, images_uint8: torch.Tensor, eta: float,
                 meta: dict | None = None) -> torch.Tensor:
        x = self.to_unit_float(images_uint8)
        self._sync(x.device)
        t0 = time.perf_counter()
        x = self.transform(x, eta, meta)
        self._sync(x.device)
        self.transform_seconds += time.perf_counter() - t0
        return self.normalizer(x)

    def describe(self) -> dict:
        return {
            "order": ["uint8/255 -> float[0,1]", "T_eta on the original image",
                      "channel normalization"],
            "transform": self.transform.describe(),
            "normalization": self.normalizer.describe(),
        }

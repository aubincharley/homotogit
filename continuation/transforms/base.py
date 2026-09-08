"""Transformation-family interface.

A *transformation family* maps an image ``x`` and a **native parameter**
``eta`` to a transformed image ``T_eta x``.  The interface deliberately makes
no assumption that the native parameter decreases toward zero at the target:
each family declares its own ``target_parameter`` and ``is_target``.  For the
Gaussian family the native parameter is ``sigma >= 0`` with the target at
``sigma = 0``; for a future relative complexity budget it would be ``t in [0,1]``
with the target at ``t = 1``.

Contract for every family implemented here:

* **Label-blind.**  ``apply`` never sees labels.
* **Deterministic** given (configuration, parameter, image) -- and given
  ``meta`` where a family needs per-image identity or calibration.
* **Applied to the original image**, never to a previously transformed one, so
  that a later, more detailed stage can recover information the earlier stage
  discarded.
* Operates on float images in ``[0, 1]`` space, *before* channel normalization,
  and never mixes colour channels.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class TransformResult:
    """Transformed images plus family-specific metadata.

    ``info`` is where a future family reports achieved complexity, feasibility
    error, solver tolerance and cost.  A *requested budget* must never be
    reported as an *achieved* one; keys are named accordingly
    (``requested_*`` vs ``achieved_*``).
    """
    images: torch.Tensor
    info: dict = field(default_factory=dict)


class ImageTransform(ABC):
    """Base class for image-transformation families."""

    #: short registry name, e.g. ``"gaussian"``
    name: str = "abstract"
    #: name of the native mathematical parameter, e.g. ``"sigma"``
    parameter_name: str = "eta"
    #: units of the native parameter, e.g. ``"pixels"`` or ``"relative budget"``
    parameter_units: str = ""

    @property
    @abstractmethod
    def target_parameter(self) -> float:
        """Native parameter value at which ``T_eta`` is the identity."""

    def is_target(self, eta: float) -> bool:
        """Whether ``eta`` is exactly the target endpoint (identity)."""
        return float(eta) == float(self.target_parameter)

    @abstractmethod
    def validate_parameter(self, eta: float) -> float:
        """Return ``eta`` as a float, raising if it is outside the valid domain."""

    @abstractmethod
    def config_signature(self) -> dict:
        """Numerical conventions and configuration that define this family.

        Everything that changes the numerical result for a *fixed* parameter
        must appear here: kernel support, padding mode, solver tolerance, and so
        on.  It is part of the cache key.
        """

    @abstractmethod
    def apply(self, x: torch.Tensor, eta: float, meta: dict | None = None) -> TransformResult:
        """Apply ``T_eta`` to a batch of float images in ``[0,1]``, ``[N,3,H,W]``."""

    def __call__(self, x: torch.Tensor, eta: float, meta: dict | None = None) -> torch.Tensor:
        return self.apply(x, eta, meta).images

    # -- identity / caching -------------------------------------------------

    def describe(self) -> dict:
        return {
            "family": self.name,
            "parameter_name": self.parameter_name,
            "parameter_units": self.parameter_units,
            "target_parameter": self.target_parameter,
            "config": self.config_signature(),
        }

    def cache_key(self, eta: float, image_id: str | int, extra: dict | None = None) -> str:
        """Unambiguous cache key: image identity + family config + parameter.

        A budget or codec *setting* is part of the key; an *achieved* complexity
        or rate is not, and must never be substituted for one.
        """
        payload = {
            "family": self.name,
            "config": self.config_signature(),
            "parameter": {self.parameter_name: float(eta)},
            "image_id": image_id,
            "extra": extra or {},
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


def check_image_batch(x: torch.Tensor, name: str = "x") -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("%s must be [N,C,H,W], got shape %s" % (name, tuple(x.shape)))
    if not x.is_floating_point():
        raise TypeError("%s must be a float tensor in [0,1] (no 8-bit quantization)" % name)
    return x

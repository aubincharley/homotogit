"""One controller for every retained method: state, per-site sigma, hooks.

The network is never duplicated and never edited.  ``attach`` registers
forward hooks at the module names given by the architecture's site map, in the
same order as the benchmark (reduction hook first, then the Gaussian hooks), and
every hook is an exact identity when its operator is inactive.

State
-----
``InterventionState(resolution, sigma)`` is what one optimizer update sees:
``resolution`` is the side length requested at the reduction point (``None``
when the method has no resolution intervention) and ``sigma`` is the scheduled
Gaussian level ``G`` before per-site scaling (``None`` when there is no
Gaussian).  The **target state** is full resolution and ``G = 0``; under it the
network is bitwise the plain model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .methods import MethodSpec
from .operators import FixedSupportGaussian, adaptive_max_reduce


class UnsupportedInsertionError(ValueError):
    """The architecture has no mapping for the requested site or reduction point."""


@dataclass(frozen=True)
class InterventionState:
    resolution: int | None
    sigma: float | None
    label: str = "custom"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None):
        return None if d is None else cls(d.get("resolution"), d.get("sigma"),
                                          d.get("label", "custom"))


class InterventionController:
    def __init__(self, method: MethodSpec, site_map: dict, arch: str = "?"):
        self.method = method
        self.site_map = site_map
        self.arch = arch
        g, r = method.gaussian, method.resolution
        if g is not None and g.placement not in site_map:
            raise UnsupportedInsertionError(
                "%s has no '%s' site map (available: %s); refusing to guess a "
                "location" % (arch, g.placement,
                              sorted(k for k in site_map if k in ("conv_out", "post_relu"))))
        if r is not None and r.point not in site_map.get("reduction", {}):
            raise UnsupportedInsertionError(
                "%s has no reduction point '%s' (available: %s); 'block1' is a "
                "ResNet-20 location and is not translated to other models"
                % (arch, r.point, sorted(site_map.get("reduction", {}))))
        self.gauss = FixedSupportGaussian(g.sigma_max, g.truncate) if g else None
        self.n_sites = len(site_map[g.placement]) if g else 0
        self.state: InterventionState | None = None

    # -- schedule ---------------------------------------------------------

    def state_for_epoch(self, epoch: int) -> InterventionState:
        r = self.method.resolution
        g = self.method.gaussian
        return InterventionState(
            resolution=int(r.schedule.at(epoch)) if r else None,
            sigma=float(g.schedule.at(epoch)) if g else None,
            label="epoch %d" % int(epoch))

    def target_state(self) -> InterventionState:
        r = self.method.resolution
        return InterventionState(resolution=r.reference_resolution if r else None,
                                 sigma=0.0 if self.method.gaussian else None,
                                 label="target")

    def set_state(self, state: InterventionState | None):
        self.state = state
        return state

    def set_epoch(self, epoch: int):
        return self.set_state(self.state_for_epoch(epoch))

    # -- per-site sigma ---------------------------------------------------

    def scale(self) -> float:
        g = self.method.gaussian
        if g is None or g.sigma_scale == "none":
            return 1.0
        r = None if self.state is None else self.state.resolution
        if r is None:
            return 1.0
        return float(r) / float(self.method.resolution.reference_resolution)

    def site_sigma(self, site: int) -> float:
        """Effective sigma at ``site`` under the current state (0 = identity)."""
        if self.gauss is None or self.state is None or self.state.sigma is None:
            return 0.0
        if not float(self.state.sigma) > 0.0:
            return 0.0
        return self.scale() * float(self.state.sigma)

    def per_site_sigma(self) -> list:
        return [self.site_sigma(i) for i in range(self.n_sites)]

    def is_target(self) -> bool:
        return self.state == self.target_state() or (
            self.state is not None
            and (self.state.sigma in (None, 0.0))
            and (self.method.resolution is None
                 or self.state.resolution == self.method.resolution.reference_resolution))

    def describe_state(self) -> dict:
        return {"state": None if self.state is None else self.state.to_dict(),
                "scale": self.scale(), "per_site_sigma": self.per_site_sigma(),
                "is_target": self.is_target()}

    # -- application ------------------------------------------------------

    def _blur(self, site: int, h: torch.Tensor) -> torch.Tensor:
        s = self.site_sigma(site)
        if s == 0.0:
            return h                                    # exact bypass
        return self.gauss(h, s)

    def _reduce(self, x: torch.Tensor):
        r = None if self.state is None else self.state.resolution
        if r is None:
            return None
        return adaptive_max_reduce(x, r)

    def attach(self, model: torch.nn.Module) -> list:
        handles = []
        r, g = self.method.resolution, self.method.gaussian
        if r is not None:
            kind, name = self.site_map["reduction"][r.point]
            mod = model.get_submodule(name)
            if kind != "input":
                raise UnsupportedInsertionError("reduction points must be module inputs")

            def _pre_reduce(_m, args):
                y = self._reduce(args[0])
                return None if y is None else (y,)
            handles.append(mod.register_forward_pre_hook(_pre_reduce))

        if g is not None:
            for site, entry in enumerate(self.site_map[g.placement]):
                kind, name = ("output", entry) if isinstance(entry, str) else entry
                mod = model.get_submodule(name)
                if kind == "output":
                    handles.append(mod.register_forward_hook(
                        lambda _m, _i, out, s=site: self._blur(s, out)))
                elif kind == "input":
                    handles.append(mod.register_forward_pre_hook(
                        lambda _m, args, s=site: (self._blur(s, args[0]),)))
                else:
                    raise UnsupportedInsertionError("unknown hook kind %r" % kind)
        return handles

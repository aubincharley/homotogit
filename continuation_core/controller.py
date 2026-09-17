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

import numpy as np

from .methods import MethodSpec
from .operators import CBSGaussian, FixedSupportGaussian, adaptive_max_reduce, sdpoint_avg_reduce
from .seeding import derive_seed


class UnsupportedInsertionError(ValueError):
    """The architecture has no mapping for the requested site or reduction point."""


@dataclass(frozen=True)
class InterventionState:
    resolution: int | None
    sigma: float | None
    label: str = "custom"
    #: SDPoint only: 1-based residual block of the downsampling point (0 = none)
    #: and its ratio; None for every other method
    sd_point: int | None = None
    sd_ratio: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.sd_point is None and self.sd_ratio is None:
            d.pop("sd_point"), d.pop("sd_ratio")      # records of the frozen methods unchanged
        return d

    @classmethod
    def from_dict(cls, d: dict | None):
        return None if d is None else cls(d.get("resolution"), d.get("sigma"),
                                          d.get("label", "custom"), d.get("sd_point"),
                                          d.get("sd_ratio"))


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
        self.cbs = CBSGaussian() if method.cbs else None
        if method.cbs:
            if method.cbs.placement not in site_map:
                raise UnsupportedInsertionError("%s has no '%s' site map" % (arch, method.cbs.placement))
            self.n_sites = len(site_map[method.cbs.placement])
        if method.sdpoint:
            if method.sdpoint.placement not in site_map:
                raise UnsupportedInsertionError("%s has no '%s' site map" % (arch, method.sdpoint.placement))
            if len(site_map[method.sdpoint.placement]) != method.sdpoint.n_points:
                raise UnsupportedInsertionError("SDPoint expects %d residual blocks, %s has %d"
                                                % (method.sdpoint.n_points, arch,
                                                   len(site_map[method.sdpoint.placement])))
        self.updates_per_epoch = None
        self.total_updates = None
        self.seed = None
        self.state: InterventionState | None = None

    def configure(self, updates_per_epoch: int, total_updates: int, seed: int):
        """Horizon and seed for update-level schedules (CBS budget-matched, SDPoint)."""
        self.updates_per_epoch, self.total_updates, self.seed = int(updates_per_epoch), int(total_updates), int(seed)
        return self

    @property
    def update_level(self) -> bool:
        return bool(self.method.sdpoint) or bool(self.method.cbs and self.method.cbs.schedule == "update_plateaus")

    # -- schedule ---------------------------------------------------------

    def state_for_epoch(self, epoch: int) -> InterventionState:
        if self.update_level:
            raise ValueError("%s has an update-level schedule: use state_for_update" % self.method.id)
        if self.method.cbs:
            return InterventionState(None, self._need().cbs.sigma_at(int(epoch) * self.updates_per_epoch,
                                                                     self.updates_per_epoch, self.total_updates),
                                     label="epoch %d" % int(epoch))
        r = self.method.resolution
        g = self.method.gaussian
        return InterventionState(
            resolution=int(r.schedule.at(epoch)) if r else None,
            sigma=float(g.schedule.at(epoch)) if g else None,
            label="epoch %d" % int(epoch))

    def _need(self):
        if self.updates_per_epoch is None:
            raise ValueError("%s needs configure(updates_per_epoch, total_updates, seed)" % self.method.id)
        return self.method

    def sdpoint_draw(self, update: int):
        """(point, ratio) of logical batch ``update``: p ~ U{0..N}, ratio ~ U(ratios)."""
        sp = self._need().sdpoint
        g = np.random.Generator(np.random.PCG64(derive_seed(self.seed, sp.stream % int(update))))
        p = int(g.integers(0, sp.n_points + 1))
        ratio = float(sp.ratios[int(g.integers(0, len(sp.ratios)))])
        return p, ratio

    def state_for_update(self, update: int, updates_per_epoch: int | None = None) -> InterventionState:
        """State used by optimizer update ``update`` (0-based).  For every
        epoch-level method this is ``state_for_epoch(update // updates_per_epoch)``."""
        upe = int(updates_per_epoch or self.updates_per_epoch)
        u = int(update)
        if self.method.sdpoint:
            p, ratio = self.sdpoint_draw(u)
            return InterventionState(None, None, label="update %d" % u, sd_point=p,
                                     sd_ratio=ratio if p > 0 else None)
        if self.method.cbs:
            m = self._need()
            return InterventionState(None, m.cbs.sigma_at(u, self.updates_per_epoch, self.total_updates),
                                     label="update %d" % u)
        return self.state_for_epoch(u // upe)

    def native_state(self) -> InterventionState:
        """Inference state of the final checkpoint in its own method's convention:
        the frozen methods' target path; CBS keeps the filter of the last update;
        SDPoint uses the full-resolution instance."""
        if self.method.cbs:
            m = self._need()
            return InterventionState(None, m.cbs.sigma_at(self.total_updates - 1, self.updates_per_epoch,
                                                           self.total_updates), label="native")
        if self.method.sdpoint:
            return InterventionState(None, None, label="native", sd_point=0)
        return InterventionState(self.target_state().resolution, self.target_state().sigma, label="native")

    def eval_current_state(self, done_epochs: int) -> InterventionState:
        """The 'current' evaluation path after ``done_epochs`` epochs: the state of
        the last update of that epoch (SDPoint: its fixed full-resolution instance)."""
        if self.method.sdpoint:
            return InterventionState(None, None, label="full-resolution instance", sd_point=0)
        if self.update_level or self.method.cbs:
            self._need()
            return self.state_for_update(max(int(done_epochs) * self.updates_per_epoch - 1, 0))
        return self.state_for_epoch(max(int(done_epochs) - 1, 0))

    def target_state(self) -> InterventionState:
        if self.method.cbs:
            return InterventionState(None, 0.0, label="target")
        if self.method.sdpoint:
            return InterventionState(None, None, label="target", sd_point=0)
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
        if self.cbs is not None:
            if self.state is None or self.state.sigma is None or not float(self.state.sigma) > 0.0:
                return 0.0
            return float(self.state.sigma)
        if self.gauss is None or self.state is None or self.state.sigma is None:
            return 0.0
        if not float(self.state.sigma) > 0.0:
            return 0.0
        return self.scale() * float(self.state.sigma)

    def per_site_sigma(self) -> list:
        return [self.site_sigma(i) for i in range(self.n_sites)]

    def is_target(self) -> bool:
        if self.method.sdpoint:
            return self.state is not None and not self.state.sd_point
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
        if self.cbs is not None:
            return self.cbs(h, s)
        return self.gauss(h, s)

    def _sdpoint(self, block: int, h: torch.Tensor) -> torch.Tensor:
        st = self.state
        if st is None or not st.sd_point or int(st.sd_point) != block + 1:
            return h                                    # exact bypass
        return sdpoint_avg_reduce(h, st.sd_ratio)

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

        if self.method.cbs is not None:
            for site, name in enumerate(self.site_map[self.method.cbs.placement]):
                handles.append(model.get_submodule(name).register_forward_hook(
                    lambda _m, _i, out, s=site: self._blur(s, out)))
        if self.method.sdpoint is not None:
            for block, name in enumerate(self.site_map[self.method.sdpoint.placement]):
                handles.append(model.get_submodule(name).register_forward_hook(
                    lambda _m, _i, out, b=block: self._sdpoint(b, out)))
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

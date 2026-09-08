r"""Continuation schedules: global step -> native transformation parameter.

A schedule is *not* a learning-rate schedule and *not* a transformation family.
It only decides which member ``T_eta`` of a family the trainer is currently
optimizing.  The learning-rate schedule lives in :mod:`continuation.optim` and is
always indexed by the global optimization step, so it never restarts at a stage
boundary.

Only :class:`ConstantSchedule` is exercised by Experiment 0.  The other
schedules are implemented and unit-tested here so that a later continuation
experiment can be configured without redesigning anything, but **no continuation
run is performed in this phase**.

Configurable independently, as required:

* ``initial`` level and ``final`` level,
* stage locations / ``steps_per_stage``,
* ``total_steps`` (the optimization budget),
* ``terminal_target_steps``: a documented terminal training period spent on the
  exact target objective (``sigma = 0`` for the Gaussian family).  A finite
  geometric sequence never reaches zero on its own, so a geometric Gaussian
  schedule *must* declare this final identity stage explicitly.

Stage boundaries also matter for the optimizer: carrying SGD momentum across a
boundary and resetting it are different procedures.  Schedules expose
``stage_index(step)`` so the trainer can implement whichever was configured;
Experiment 0 has no stage transitions at all.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

from .config import ScheduleConfig


class Schedule(ABC):
    """Maps global optimization step to a native transformation parameter."""

    kind: str = "abstract"
    #: True when the parameter changes in discrete stages
    stagewise: bool = False

    def __init__(self, total_steps: int):
        self.total_steps = int(total_steps)

    @abstractmethod
    def parameter(self, step: int, state: dict | None = None) -> float:
        """Native parameter at ``step`` (0-based).  ``state`` is reserved for
        future adaptive controllers and is ignored by every schedule here."""

    def stage_index(self, step: int) -> int | None:
        """Stage number at ``step``, or ``None`` for smoothly varying schedules."""
        return None

    def progress(self, step: int) -> float:
        if self.total_steps <= 1:
            return 0.0
        return min(max(step / float(self.total_steps - 1), 0.0), 1.0)

    def describe(self) -> dict:
        return {"kind": self.kind, "stagewise": self.stagewise,
                "total_steps": self.total_steps}


class ConstantSchedule(Schedule):
    """A fixed transformation level for the whole run (Experiment 0)."""

    kind = "constant"

    def __init__(self, total_steps: int, value: float):
        super().__init__(total_steps)
        self.value = float(value)

    def parameter(self, step: int, state: dict | None = None) -> float:
        return self.value

    def stage_index(self, step: int) -> int:
        return 0

    def describe(self) -> dict:
        d = super().describe()
        d["value"] = self.value
        return d


class PiecewiseConstantSchedule(Schedule):
    """Piecewise-constant levels with an explicit update count per stage.

    ``values`` and ``steps_per_stage`` have the same length; the last stage
    absorbs any remaining steps so the parameter is defined for every step of
    the budget.  Put the target endpoint last (and size that stage with
    ``terminal_target_steps``) to allocate a terminal period to the exact target
    objective.
    """

    kind = "piecewise_constant"
    stagewise = True

    def __init__(self, total_steps: int, values, steps_per_stage):
        super().__init__(total_steps)
        values = [float(v) for v in values]
        steps = [int(s) for s in steps_per_stage]
        if len(values) != len(steps):
            raise ValueError("values and steps_per_stage must have equal length")
        if not values:
            raise ValueError("at least one stage is required")
        if any(s <= 0 for s in steps):
            raise ValueError("steps_per_stage entries must be positive")
        self.values = values
        self.steps_per_stage = steps
        self.boundaries = []
        acc = 0
        for s in steps:
            acc += s
            self.boundaries.append(acc)
        self.planned_steps = acc

    def stage_index(self, step: int) -> int:
        for i, b in enumerate(self.boundaries):
            if step < b:
                return i
        return len(self.values) - 1

    def parameter(self, step: int, state: dict | None = None) -> float:
        return self.values[self.stage_index(step)]

    def describe(self) -> dict:
        d = super().describe()
        d.update(values=self.values, steps_per_stage=self.steps_per_stage,
                 boundaries=self.boundaries, planned_steps=self.planned_steps,
                 note=("last stage absorbs any steps beyond planned_steps"
                       if self.planned_steps < self.total_steps else ""))
        return d


class LinearSigmaSchedule(Schedule):
    """``sigma`` linear in training progress, from ``start`` to ``end``."""

    kind = "linear_sigma"

    def __init__(self, total_steps: int, start: float, end: float = 0.0):
        super().__init__(total_steps)
        self.start, self.end = float(start), float(end)

    def parameter(self, step: int, state: dict | None = None) -> float:
        p = self.progress(step)
        return self.start + (self.end - self.start) * p

    def describe(self) -> dict:
        d = super().describe()
        d.update(start=self.start, end=self.end)
        return d


class GeometricSigmaSchedule(Schedule):
    """Geometric decay of ``sigma`` with an **explicit final identity stage**.

    A finite geometric sequence never reaches zero, so the target endpoint is
    reached only because the last ``terminal_target_steps`` updates are pinned to
    ``target`` (``sigma = 0``).  The geometric part runs over the remaining
    ``total_steps - terminal_target_steps`` updates, in ``num_stages``
    piecewise-constant stages by default, or continuously when
    ``num_stages is None``.
    """

    kind = "geometric_sigma"

    def __init__(self, total_steps: int, start: float, end: float,
                 terminal_target_steps: int, num_stages: int | None = None,
                 target: float = 0.0):
        super().__init__(total_steps)
        if start <= 0 or end <= 0:
            raise ValueError("geometric schedule needs start > 0 and end > 0; "
                             "the exact target endpoint is reached by the explicit "
                             "terminal stage, not by the geometric sequence")
        if terminal_target_steps <= 0:
            raise ValueError("terminal_target_steps must be > 0: a geometric sequence "
                             "does not reach the target endpoint by itself")
        if terminal_target_steps >= total_steps:
            raise ValueError("terminal_target_steps must be < total_steps")
        if num_stages is not None and num_stages < 1:
            raise ValueError("num_stages must be >= 1 or None")
        self.start, self.end = float(start), float(end)
        self.terminal_target_steps = int(terminal_target_steps)
        self.num_stages = num_stages
        self.target = float(target)
        self.geometric_steps = self.total_steps - self.terminal_target_steps
        self.stagewise = num_stages is not None

    def stage_index(self, step: int) -> int:
        if step >= self.geometric_steps:
            return -1 if self.num_stages is None else self.num_stages
        if self.num_stages is None:
            return 0
        frac = step / float(self.geometric_steps)
        return min(int(frac * self.num_stages), self.num_stages - 1)

    def parameter(self, step: int, state: dict | None = None) -> float:
        if step >= self.geometric_steps:
            return self.target
        log_s, log_e = math.log(self.start), math.log(self.end)
        if self.num_stages is None:
            denom = max(self.geometric_steps - 1, 1)
            frac = step / float(denom)
        else:
            frac = (self.stage_index(step) / float(self.num_stages - 1)
                    if self.num_stages > 1 else 0.0)
        return math.exp(log_s + (log_e - log_s) * frac)

    def describe(self) -> dict:
        d = super().describe()
        d.update(start=self.start, end=self.end, target=self.target,
                 num_stages=self.num_stages,
                 terminal_target_steps=self.terminal_target_steps,
                 geometric_steps=self.geometric_steps,
                 note="explicit final identity stage; the geometric part never reaches the target")
        return d


class HeatTimeSchedule(Schedule):
    r"""Linear or geometric in heat time ``a = sigma^2 / 2``; returns ``sigma``.

    Motivated by the *continuous* heat semigroup.  The discrete, truncated,
    padded filter does not satisfy that semigroup identity exactly, so this is a
    parameterization choice, not an equivalence claim.
    """

    kind = "heat_time"

    def __init__(self, total_steps: int, start_sigma: float, end_sigma: float,
                 mode: str = "linear", terminal_target_steps: int = 0):
        super().__init__(total_steps)
        if mode not in ("linear", "geometric"):
            raise ValueError("mode must be 'linear' or 'geometric', got %r" % (mode,))
        if mode == "geometric" and (start_sigma <= 0 or end_sigma <= 0):
            raise ValueError("geometric heat-time schedule needs positive sigmas; "
                             "use terminal_target_steps for the exact endpoint")
        if terminal_target_steps < 0 or terminal_target_steps >= total_steps:
            raise ValueError("terminal_target_steps must be in [0, total_steps)")
        self.start_sigma, self.end_sigma = float(start_sigma), float(end_sigma)
        self.mode = mode
        self.terminal_target_steps = int(terminal_target_steps)
        self.active_steps = self.total_steps - self.terminal_target_steps

    def parameter(self, step: int, state: dict | None = None) -> float:
        if step >= self.active_steps:
            return 0.0
        a0 = self.start_sigma ** 2 / 2.0
        a1 = self.end_sigma ** 2 / 2.0
        denom = max(self.active_steps - 1, 1)
        p = min(max(step / float(denom), 0.0), 1.0)
        if self.mode == "linear":
            a = a0 + (a1 - a0) * p
        else:
            a = math.exp(math.log(a0) + (math.log(a1) - math.log(a0)) * p)
        return math.sqrt(max(2.0 * a, 0.0))

    def describe(self) -> dict:
        d = super().describe()
        d.update(start_sigma=self.start_sigma, end_sigma=self.end_sigma, mode=self.mode,
                 terminal_target_steps=self.terminal_target_steps,
                 note="a = sigma^2/2; continuous-semigroup motivation only")
        return d


class IncreasingBudgetSchedule(Schedule):
    """Increasing relative complexity budget ``t: t_start -> 1`` for future TV runs.

    Included so that a family whose target endpoint is at ``t = 1`` (rather than
    at zero) is representable today.  No TV transformation exists yet, so this
    schedule currently has no family to drive.
    """

    kind = "increasing_budget"

    def __init__(self, total_steps: int, start: float = 0.2, end: float = 1.0,
                 mode: str = "linear", terminal_target_steps: int = 0):
        super().__init__(total_steps)
        if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
            raise ValueError("relative budgets must lie in [0,1]")
        if mode not in ("linear", "geometric"):
            raise ValueError("mode must be 'linear' or 'geometric'")
        if mode == "geometric" and start <= 0:
            raise ValueError("geometric budget schedule needs start > 0")
        if terminal_target_steps < 0 or terminal_target_steps >= total_steps:
            raise ValueError("terminal_target_steps must be in [0, total_steps)")
        self.start, self.end, self.mode = float(start), float(end), mode
        self.terminal_target_steps = int(terminal_target_steps)
        self.active_steps = self.total_steps - self.terminal_target_steps

    def parameter(self, step: int, state: dict | None = None) -> float:
        if step >= self.active_steps:
            return 1.0
        denom = max(self.active_steps - 1, 1)
        p = min(max(step / float(denom), 0.0), 1.0)
        if self.mode == "linear":
            return self.start + (self.end - self.start) * p
        return math.exp(math.log(self.start) + (math.log(self.end) - math.log(self.start)) * p)

    def describe(self) -> dict:
        d = super().describe()
        d.update(start=self.start, end=self.end, mode=self.mode,
                 terminal_target_steps=self.terminal_target_steps)
        return d


_BUILDERS = {
    "constant": ConstantSchedule,
    "piecewise_constant": PiecewiseConstantSchedule,
    "linear_sigma": LinearSigmaSchedule,
    "geometric_sigma": GeometricSigmaSchedule,
    "heat_time": HeatTimeSchedule,
    "increasing_budget": IncreasingBudgetSchedule,
}


def available_schedules():
    return sorted(_BUILDERS)


def build_schedule(cfg: ScheduleConfig, total_steps: int) -> Schedule:
    if cfg.kind == "adaptive":
        raise NotImplementedError(
            "adaptive schedules driven by optimization state are deliberately not "
            "implemented in this phase (see docs/experiment0.md, 'what is out of scope')"
        )
    if cfg.kind not in _BUILDERS:
        raise KeyError("unknown schedule %r; available: %s" % (cfg.kind, available_schedules()))
    params = dict(cfg.params or {})
    # Consumed by the trainer, not by the schedule: carrying vs resetting SGD
    # momentum at a stage boundary is an optimizer policy that happens to be
    # configured next to the schedule it applies to.
    params.pop("momentum_at_stage_boundary", None)
    return _BUILDERS[cfg.kind](total_steps=total_steps, **params)

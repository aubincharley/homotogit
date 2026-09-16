"""Piecewise-constant schedules indexed by zero-based epoch.

A value holds for the **whole** epoch: every optimizer update of epoch ``e``
uses ``schedule.at(e)``.  Transitions therefore happen exactly at epoch
boundaries, i.e. between update ``starts[i] * updates_per_epoch - 1`` and the
next one.  Epochs past the last boundary keep the last value.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass


@dataclass(frozen=True)
class EpochSchedule:
    starts: tuple
    values: tuple

    def __post_init__(self):
        starts, values = tuple(int(s) for s in self.starts), tuple(self.values)
        if not starts or starts[0] != 0:
            raise ValueError("a schedule must start at epoch 0, got %s" % (starts,))
        if any(b <= a for a, b in zip(starts, starts[1:])):
            raise ValueError("schedule starts must strictly increase: %s" % (starts,))
        if len(starts) != len(values):
            raise ValueError("%d starts but %d values" % (len(starts), len(values)))
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "values", values)

    def at(self, epoch: int):
        i = bisect.bisect_right(self.starts, max(int(epoch), 0)) - 1
        return self.values[i]

    def table(self, epochs: int) -> list:
        return [self.at(e) for e in range(int(epochs))]

    def transitions(self) -> tuple:
        """Epochs at which the value changes (the first epoch using the new value)."""
        return tuple(s for i, s in enumerate(self.starts)
                     if i > 0 and self.values[i] != self.values[i - 1])

    def to_dict(self) -> dict:
        return {"starts": list(self.starts), "values": list(self.values)}

    @classmethod
    def from_dict(cls, d: dict) -> "EpochSchedule":
        return cls(tuple(d["starts"]), tuple(d["values"]))

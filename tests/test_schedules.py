import math

import pytest

from continuation.config import ScheduleConfig
from continuation.schedules import (
    ConstantSchedule,
    GeometricSigmaSchedule,
    HeatTimeSchedule,
    IncreasingBudgetSchedule,
    LinearSigmaSchedule,
    PiecewiseConstantSchedule,
    build_schedule,
)


def test_constant_schedule_is_the_fixed_level_case():
    s = ConstantSchedule(1000, 2.0)
    assert [s.parameter(i) for i in (0, 500, 999)] == [2.0, 2.0, 2.0]
    assert s.stage_index(0) == 0


def test_piecewise_constant_boundaries_and_stage_indices():
    s = PiecewiseConstantSchedule(1000, values=[3.0, 1.0, 0.0], steps_per_stage=[400, 400, 200])
    assert s.parameter(0) == 3.0 and s.parameter(399) == 3.0
    assert s.parameter(400) == 1.0 and s.parameter(799) == 1.0
    assert s.parameter(800) == 0.0 and s.parameter(999) == 0.0
    assert [s.stage_index(i) for i in (0, 400, 800)] == [0, 1, 2]
    assert s.parameter(5000) == 0.0          # last stage absorbs overrun


def test_piecewise_rejects_inconsistent_input():
    with pytest.raises(ValueError):
        PiecewiseConstantSchedule(100, [1.0, 0.0], [50])
    with pytest.raises(ValueError):
        PiecewiseConstantSchedule(100, [1.0], [0])


def test_linear_sigma_hits_both_endpoints():
    s = LinearSigmaSchedule(101, start=3.0, end=0.0)
    assert s.parameter(0) == pytest.approx(3.0)
    assert s.parameter(50) == pytest.approx(1.5)
    assert s.parameter(100) == pytest.approx(0.0)


def test_geometric_sigma_requires_and_uses_an_explicit_identity_stage():
    with pytest.raises(ValueError):
        GeometricSigmaSchedule(1000, start=3.0, end=0.0, terminal_target_steps=100)
    with pytest.raises(ValueError):
        GeometricSigmaSchedule(1000, start=3.0, end=0.5, terminal_target_steps=0)

    s = GeometricSigmaSchedule(1000, start=3.0, end=0.5, terminal_target_steps=200, num_stages=4)
    assert s.parameter(0) == pytest.approx(3.0)
    assert s.parameter(799) == pytest.approx(0.5)
    # the geometric part never reaches the target; the terminal stage does
    assert all(s.parameter(i) > 0 for i in range(800))
    assert s.parameter(800) == 0.0 and s.parameter(999) == 0.0
    values = sorted({round(s.parameter(i), 9) for i in range(800)}, reverse=True)
    assert len(values) == 4
    ratios = [values[i] / values[i + 1] for i in range(3)]
    assert all(r == pytest.approx(ratios[0]) for r in ratios)   # geometric


def test_geometric_sigma_continuous_variant():
    s = GeometricSigmaSchedule(1000, start=3.0, end=0.5, terminal_target_steps=200,
                               num_stages=None)
    assert s.stagewise is False
    assert s.parameter(0) == pytest.approx(3.0)
    assert s.parameter(799) == pytest.approx(0.5)
    seq = [s.parameter(i) for i in range(0, 800, 50)]
    assert all(a > b for a, b in zip(seq, seq[1:]))


def test_heat_time_schedule_is_linear_in_a_not_in_sigma():
    s = HeatTimeSchedule(101, start_sigma=3.0, end_sigma=0.0, mode="linear")
    a0 = 3.0 ** 2 / 2
    assert s.parameter(0) == pytest.approx(3.0)
    assert s.parameter(100) == pytest.approx(0.0)
    mid = s.parameter(50)
    assert mid ** 2 / 2 == pytest.approx(a0 / 2, rel=1e-6)
    assert mid != pytest.approx(1.5)          # would be the sigma-linear midpoint


def test_heat_time_geometric_needs_positive_endpoints_and_a_terminal_stage():
    with pytest.raises(ValueError):
        HeatTimeSchedule(100, 3.0, 0.0, mode="geometric")
    s = HeatTimeSchedule(1000, 3.0, 0.5, mode="geometric", terminal_target_steps=100)
    assert s.parameter(899) > 0 and s.parameter(900) == 0.0
    a = [s.parameter(i) ** 2 / 2 for i in (0, 300, 600, 899)]
    assert all(x > y for x, y in zip(a, a[1:]))


def test_increasing_budget_targets_one_not_zero():
    s = IncreasingBudgetSchedule(101, start=0.2, end=1.0)
    assert s.parameter(0) == pytest.approx(0.2)
    assert s.parameter(100) == pytest.approx(1.0)
    seq = [s.parameter(i) for i in range(0, 101, 10)]
    assert all(a < b for a, b in zip(seq, seq[1:]))


def test_builder_and_adaptive_is_explicitly_not_implemented():
    s = build_schedule(ScheduleConfig(kind="constant", params={"value": 1.5}), 100)
    assert isinstance(s, ConstantSchedule) and s.parameter(10) == 1.5
    with pytest.raises(NotImplementedError):
        build_schedule(ScheduleConfig(kind="adaptive", params={}), 100)
    with pytest.raises(KeyError):
        build_schedule(ScheduleConfig(kind="nope", params={}), 100)

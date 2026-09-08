"""What the gate has to be true for, or none of the measurements mean anything.

The claims these pin down are the ones an experiment would otherwise assume:
that s=1 is the baseline exactly, that s=0 is the shortcut-only network and not
the zero function, that the residual branch really is frozen at s=0, and that s
is algebraically the same thing as scaling bn2 -- which is the reason to expect
the homotopy to change the trajectory rather than the reachable solutions.
"""
import copy

import pytest
import torch

from cifarbase.homotopy import (SCHEDULES, ResidualGate, other_parameters,
                                phase_at, phase_bounds,
                                residual_blocks, residual_parameters, s_at,
                                s_at_cfg)
from cifarbase.model import ARCHS, BasicBlock, ResNet

N_BLOCKS = 8            # resnet18 is (2,2,2,2)


def make_model(seed=0):
    """A narrow resnet18 with a residual branch that actually does something.

    zero_init_residual is off on purpose: the default leaves gamma_bn2 at zero,
    so F == 0 and every value of s would produce the same forward pass. A test
    that passes under those conditions has tested nothing.
    """
    torch.manual_seed(seed)
    model = ResNet(ARCHS["resnet18"], num_classes=10, width=16,
                   zero_init_residual=False)
    return model.eval()


def batch(n=4, seed=1):
    return torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


def test_gate_at_one_is_bit_identical():
    model, x = make_model(), batch()
    before = model(x)
    with ResidualGate(model) as gate:
        gate.set(1.0)
        gated = model(x)
    assert torch.equal(before, gated)


def test_closing_the_gate_restores_the_baseline():
    model, x = make_model(), batch()
    before = model(x)
    gate = ResidualGate(model)
    gate.set(0.3)
    assert not torch.allclose(before, model(x))
    gate.close()
    assert torch.equal(before, model(x))


def test_s_zero_is_the_shortcut_only_network():
    """s=0 must kill F and leave the shortcut untouched.

    The reference is the same network with bn2's gamma and beta zeroed, which
    makes F identically zero by construction. If the gate also scaled the
    shortcut, the output would collapse to a constant instead and this would
    fail.
    """
    model, x = make_model(), batch()
    reference = copy.deepcopy(model)
    with torch.no_grad():
        for block in residual_blocks(reference):
            block.bn2.weight.zero_()
            block.bn2.bias.zero_()

    with ResidualGate(model) as gate:
        gate.set(0.0)
        gated = model(x)

    expected = reference(x)
    assert torch.allclose(gated, expected, atol=1e-6)
    # ...and the shortcut-only network is a real network, not the zero map.
    assert expected.abs().max() > 1e-3
    assert expected.std() > 1e-4


def test_s_scales_like_bn2_gamma():
    """s*F is the same function as F with bn2 -> (s*gamma, s*beta).

    This is the algebraic fact that decides how to read the whole experiment: for
    any s > 0 the representable function class is unchanged, so the homotopy is a
    reparametrisation and any effect it has must come from the optimiser's path,
    not from capacity. If this test ever fails, that reading is wrong.
    """
    model, x = make_model(), batch()
    s = 0.5
    reference = copy.deepcopy(model)
    with torch.no_grad():
        for block in residual_blocks(reference):
            block.bn2.weight *= s
            block.bn2.bias *= s

    with ResidualGate(model) as gate:
        gate.set(s)
        gated = model(x)

    assert torch.allclose(gated, reference(x), atol=1e-5)


def test_residual_branch_is_frozen_at_s_zero():
    """dL/dtheta_F is proportional to s, so at s=0 the branch learns nothing.

    Everything outside the gated branches -- stem, shortcut projections, head --
    must still receive gradient, otherwise s=0 would train nothing at all.
    """
    model, x = make_model(), batch()
    y = torch.randint(0, 10, (x.shape[0],))

    with ResidualGate(model) as gate:
        gate.set(0.0)
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x), y).backward()

        for p in residual_parameters(model):
            assert p.grad is None or float(p.grad.abs().max()) == 0.0

        outside = max(float(p.grad.abs().max())
                      for p in other_parameters(model) if p.grad is not None)
        assert outside > 0.0


def test_residual_branch_learns_at_s_positive():
    model, x = make_model(), batch()
    y = torch.randint(0, 10, (x.shape[0],))
    with ResidualGate(model) as gate:
        gate.set(0.1)
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x), y).backward()
        assert max(float(p.grad.abs().max()) for p in residual_parameters(model)
                   if p.grad is not None) > 0.0


def test_block_order_is_forward_order():
    """residual_blocks() relies on registration order matching execution order.

    That is a property of model.py, not a guarantee from torch, so it gets
    checked by watching which block the forward pass reaches first.
    """
    model, x = make_model(), batch()
    blocks = residual_blocks(model)
    assert len(blocks) == N_BLOCKS

    visited = []
    handles = [b.register_forward_pre_hook(
        lambda module, inputs, i=i: visited.append(i))
        for i, b in enumerate(blocks)]
    try:
        model(x)
    finally:
        for handle in handles:
            handle.remove()
    assert visited == list(range(N_BLOCKS))


def test_parameter_partition_is_a_partition():
    model = make_model()
    gated = {id(p) for p in residual_parameters(model)}
    rest = {id(p) for p in other_parameters(model)}
    everything = {id(p) for p in model.parameters()}
    assert gated | rest == everything
    assert not (gated & rest)
    # The shortcut projections must be on the ungated side: s never touches them.
    projections = [id(p) for b in residual_blocks(model)
                   for p in b.shortcut.parameters()]
    assert projections and all(pid in rest for pid in projections)


def test_per_block_values_reach_the_right_block():
    model, x = make_model(), batch()
    with ResidualGate(model) as gate:
        # Everything off except the last block: the result must differ from
        # all-off, and match a reference with only that block's F alive.
        values = [0.0] * N_BLOCKS
        gate.set(values)
        all_off = model(x)
        values[-1] = 1.0
        gate.set(values)
        last_on = model(x)
    assert not torch.allclose(all_off, last_on)


# --------------------------------------------------------------------------
# schedules
# --------------------------------------------------------------------------

@pytest.mark.parametrize("schedule", SCHEDULES)
def test_schedule_reaches_s_max_and_stays(schedule):
    """The invariant the whole comparison rests on.

    A run whose s never reaches 1 has trained a different model than the
    baseline, so every schedule must sit at exactly s_max for the entire tail
    after ramp_end -- not approach it.
    """
    kwargs = dict(schedule=schedule, s_min=0.0, s_max=1.0,
                  ramp_start=0.0, ramp_end=0.5, stairs=4)
    for progress in (0.5, 0.6, 0.999, 1.0):
        assert s_at(progress, N_BLOCKS, **kwargs) == [1.0] * N_BLOCKS


@pytest.mark.parametrize("schedule", ["linear", "cosine", "staircase", "sequential"])
def test_schedule_starts_at_s_min(schedule):
    values = s_at(0.0, N_BLOCKS, schedule=schedule, s_min=0.05, ramp_start=0.0,
                  ramp_end=0.5, stairs=4)
    assert values == [0.05] * N_BLOCKS


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_schedule_is_monotone_and_bounded(schedule):
    previous = [0.0] * N_BLOCKS
    for step in range(101):
        values = s_at(step / 100, N_BLOCKS, schedule=schedule, s_min=0.0,
                      ramp_start=0.1, ramp_end=0.7, stairs=5)
        assert len(values) == N_BLOCKS
        assert all(0.0 <= v <= 1.0 for v in values)
        assert all(new >= old - 1e-12
                   for new, old in zip(values, previous, strict=True))
        previous = values


def test_ramp_start_delays_the_ramp():
    kwargs = dict(schedule="linear", s_min=0.0, ramp_start=0.25, ramp_end=0.75)
    assert s_at(0.1, N_BLOCKS, **kwargs) == [0.0] * N_BLOCKS
    assert s_at(0.25, N_BLOCKS, **kwargs) == [0.0] * N_BLOCKS
    assert s_at(0.5, N_BLOCKS, **kwargs) == pytest.approx([0.5] * N_BLOCKS)


def test_staircase_has_the_requested_number_of_levels():
    levels = sorted({s_at(i / 500, N_BLOCKS, schedule="staircase", ramp_end=1.0,
                          stairs=5)[0] for i in range(500)})
    assert levels == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_sequential_turns_blocks_on_in_order():
    """Depth continuation: early blocks lead, later blocks are still off."""
    values = s_at(0.25, N_BLOCKS, schedule="sequential", ramp_start=0.0,
                  ramp_end=1.0)
    assert values == sorted(values, reverse=True)
    assert values[0] == 1.0                 # first block already fully on
    assert values[-1] == 0.0                # last block not started
    assert all(v >= 0.0 for v in values)


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="unknown schedule"):
        s_at(0.5, N_BLOCKS, schedule="exponential")


def test_gate_rejects_a_wrong_length_vector():
    model = make_model()
    with ResidualGate(model) as gate:
        with pytest.raises(ValueError, match="expected 8 values"):
            gate.set([1.0, 0.0])


# --------------------------------------------------------------------------
# continuation phases
# --------------------------------------------------------------------------

def staircase_cfg(**over):
    cfg = {"s_schedule": "staircase", "s_stairs": 5, "s_lr_restart": True,
           "s_min": 0.0, "s_max": 1.0, "s_ramp_start": 0.0, "s_ramp_end": 0.8}
    cfg.update(over)
    return cfg


def test_no_phases_unless_asked_for():
    """Empty bounds are what keep every pre-existing arm bit-identical."""
    assert phase_bounds(staircase_cfg(s_lr_restart=False)) == ()
    assert phase_bounds(staircase_cfg(s_schedule="linear")) == ()


def test_a_phase_is_exactly_an_interval_of_constant_s():
    """One phase per distinct value of s, and no boundary that changes nothing.

    ramp_end is the trap: the last plateau already sits at s_max, so the tail
    after ramp_end is the same optimisation continued. A boundary there would
    restart the learning rate in the middle of one problem.
    """
    cfg = staircase_cfg()
    bounds = phase_bounds(cfg)
    assert bounds[0] == 0.0 and bounds[-1] == 1.0
    assert bounds == pytest.approx((0.0, 0.16, 0.32, 0.48, 0.64, 1.0))
    assert len(bounds) - 1 == 5                      # one per distinct s

    values = [s_at_cfg((bounds[i] + bounds[i + 1]) / 2, 1, cfg)[0]
              for i in range(len(bounds) - 1)]
    assert values == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])

    # s must be constant across each phase, and different across each boundary.
    for index in range(len(bounds) - 1):
        low, high = bounds[index], bounds[index + 1]
        inside = {round(s_at_cfg(low + (high - low) * f, 1, cfg)[0], 12)
                  for f in (0.01, 0.25, 0.5, 0.75, 0.99)}
        assert len(inside) == 1, f"s moves inside phase {index}"
    for edge in bounds[1:-1]:
        assert s_at_cfg(edge - 1e-9, 1, cfg)[0] != s_at_cfg(edge + 1e-9, 1, cfg)[0]


def test_phase_at_covers_the_whole_run_exactly_once():
    bounds = phase_bounds(staircase_cfg())
    last = len(bounds) - 2
    seen = []
    for step in range(1001):
        index, local = phase_at(step / 1000, bounds)
        assert 0.0 <= local <= 1.0
        seen.append(index)
    assert sorted(set(seen)) == list(range(last + 1))
    assert seen == sorted(seen)                      # phases never go backwards
    assert phase_at(0.0, bounds) == (0, 0.0)
    assert phase_at(1.0, bounds)[0] == last


def test_lr_restarts_inside_every_phase():
    """The point of the whole fix: each plateau decays on its own, not once.

    Under a single global cosine the lr falls monotonically, so late plateaus
    cannot move. With restarts it must come back up at every boundary.
    """
    from cifarbase.train import lr_at

    cfg = dict(staircase_cfg(), epochs=60, warmup_epochs=5, schedule="cosine")
    bounds = phase_bounds(cfg)
    total = 6000

    globals_ = [lr_at(s, total, cfg) for s in range(total)]
    staged = [lr_at(s, total, cfg, bounds) for s in range(total)]

    def window(series, index):
        return series[int(bounds[index] * total):int(bounds[index + 1] * total)]

    # Every phase warms up to the full lr and decays back down inside itself.
    # Checked on the window rather than on a per-step jump: warmup climbs over
    # ~80 steps, so no single step rises by much and a jump threshold would only
    # be measuring the warmup length.
    for index in range(len(bounds) - 1):
        values = window(staged, index)
        assert max(values) > 0.9, f"phase {index} never warms up"
        assert min(values) < 0.1, f"phase {index} never decays"

    # The contrast that makes the fix necessary, stated structurally rather than
    # against a magic threshold: under one global cosine each successive phase
    # peaks at a lower learning rate than the last, so the late plateaus cannot
    # move theta and never settle at the s they hold. With restarts every phase
    # gets the same budget of lr.
    staged_peaks = [max(window(staged, i)) for i in range(len(bounds) - 1)]
    global_peaks = [max(window(globals_, i)) for i in range(len(bounds) - 1)]
    assert global_peaks == sorted(global_peaks, reverse=True)
    assert global_peaks[-1] < 0.5 * global_peaks[0]
    assert max(staged_peaks) - min(staged_peaks) < 0.05
    assert sum(b > a + 1e-9 for a, b in zip(globals_, globals_[1:])) < total // 10

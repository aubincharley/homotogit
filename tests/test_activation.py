"""What the activation homotopy has to be true for.

phi_alpha(x) = max(x, alpha*x), driven 1 -> 0. The claims pinned down here are
the ones an experiment would otherwise assume: that alpha=0 is the baseline
ReLU network *exactly*, that alpha=1 really does make the network affine, that
alpha cannot be reparametrised away the way s can, and that the per-site alpha
vector reaches the sites the scope says it should.
"""
import copy

import pytest
import torch
import torch.nn.functional as F

from cifarbase.activation import (SCOPES, SITES, ActivationGate,
                                  activation_sites, build_activation_gate,
                                  site_groups)
from cifarbase.config import load_config
from cifarbase.train import readout_at_alpha_zero
from cifarbase.homotopy import alpha_at_cfg, phase_bounds
from cifarbase.data import Split
from cifarbase.landscape import (activation_stats, evaluate_at,
                                 functional_distance, get_weights,
                                 loss_vs_alpha, random_direction,
                                 residual_ratio)
from cifarbase.model import ARCHS, Activation, ResNet

N_SITES = 17            # resnet18: the stem, plus two per block, eight blocks
N_KEYS = 122            # state_dict entries, measured before Activation existed


def make_model(seed=0, width=16, zero_init_residual=False):
    """A narrow resnet18 whose residual branch actually does something.

    zero_init_residual is off on purpose, for the same reason test_homotopy
    turns it off: the default leaves gamma_bn2 at zero, so F == 0 and the
    post-add activation sees only the shortcut. A test that passes under those
    conditions has tested half the network.
    """
    torch.manual_seed(seed)
    model = ResNet(ARCHS["resnet18"], num_classes=10, width=width,
                   zero_init_residual=zero_init_residual)
    return model.eval()


def batch(n=4, seed=1):
    return torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


def relu_forward(model, x):
    """ResNet.forward as it was written before Activation existed.

    The reference the alpha=0 path has to reproduce bit for bit. Written out
    here rather than imported so that a change to model.py cannot silently
    move the thing being compared against.
    """
    out = F.relu(model.bn1(model.conv1(x)))
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in stage:
            inner = F.relu(block.bn1(block.conv1(out)))
            inner = block.bn2(block.conv2(inner))
            out = F.relu(inner + block.shortcut(out))
    out = F.adaptive_avg_pool2d(out, 1).flatten(1)
    return model.fc(out)


# --------------------------------------------------------------------------
# the activation itself
# --------------------------------------------------------------------------

def test_activation_at_alpha_zero_is_exactly_relu():
    act, x = Activation(inplace=False), torch.randn(64)
    assert torch.equal(act(x), F.relu(x))


def test_activation_at_alpha_one_is_the_identity():
    act, x = Activation(inplace=False), torch.randn(64)
    act.alpha = 1.0
    assert torch.equal(act(x), x)


def test_activation_interpolates_max_of_x_and_alpha_x():
    act, x = Activation(inplace=False), torch.randn(64)
    act.alpha = 0.37
    assert torch.allclose(act(x), torch.maximum(x, 0.37 * x))


def test_activation_is_positively_homogeneous():
    """phi(c*x) = c*phi(x) for c > 0 -- the property that closes the escape.

    s could always be absorbed into bn2's gamma, so a residual-homotopy run can
    reparametrise s away. alpha cannot: no rescaling of any weight changes the
    mix of x and |x| that phi_alpha applies. This is the reason to expect the
    activation homotopy to deform the network rather than its parametrisation.
    """
    act, x = Activation(inplace=False), torch.randn(64)
    act.alpha = 0.4
    assert torch.allclose(act(3.7 * x), 3.7 * act(x), atol=1e-6)


# --------------------------------------------------------------------------
# the model wearing them
# --------------------------------------------------------------------------

def test_model_at_alpha_zero_is_bit_identical_to_the_relu_forward():
    model, x = make_model(), batch()
    assert torch.equal(model(x), relu_forward(model, x))


def test_gradients_at_alpha_zero_are_bit_identical_too():
    """Equal outputs are not enough: the baseline has to be the same run.

    A forward that agrees but a backward that does not would leave every arm
    comparable on paper and different in fact.
    """
    model, x = make_model(), batch()
    model(x).square().sum().backward()
    ours = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    relu_forward(model, x).square().sum().backward()
    assert all(torch.equal(a, p.grad) for a, p in zip(ours, model.parameters()))


def test_activation_modules_add_no_state():
    """No parameters, no buffers, so a checkpoint written before this class
    existed still loads, and a homotopy model shares a state_dict with a
    baseline one -- the invariant ResidualGate maintains for s."""
    model = make_model()
    sites = [m for m in model.modules() if isinstance(m, Activation)]
    assert len(sites) == N_SITES
    assert not any(list(m.parameters()) or list(m.buffers()) for m in sites)
    assert len(model.state_dict()) == N_KEYS
    assert not any("act" in key for key in model.state_dict())


def test_activation_modules_add_no_parameters():
    model = make_model()
    assert sum(p.numel() for p in model.parameters()) == 701_466  # width 16, measured before Activation existed


def test_model_at_alpha_one_is_affine():
    """The claim the whole homotopy rests on: alpha=1 is a linear network.

    Affine means f(x+y) - f(x) - f(y) + f(0) == 0. Checked in eval mode, where
    BatchNorm is an affine map of its input rather than a function of the batch.
    """
    model = make_model()
    for site in model.modules():
        if isinstance(site, Activation):
            site.alpha = 1.0
    x, y = batch(seed=1), batch(seed=2)
    residual = model(x + y) - model(x) - model(y) + model(torch.zeros_like(x))
    assert residual.abs().max() < 1e-3 * model(x).abs().max()


def test_model_at_alpha_one_is_not_affine_at_alpha_zero():
    """The control for the test above: the same check has to fail for ReLU, or
    it is measuring float tolerance rather than linearity."""
    model = make_model()
    x, y = batch(seed=1), batch(seed=2)
    residual = model(x + y) - model(x) - model(y) + model(torch.zeros_like(x))
    assert residual.abs().max() > 1e-3 * model(x).abs().max()


# --------------------------------------------------------------------------
# the sites, and the scopes that bundle them
# --------------------------------------------------------------------------

def test_sites_are_every_activation_in_forward_order():
    """Registration order happens to be forward order for this model, which is
    a fact about model.py rather than a promise from torch. Hooks pin it down
    instead of trusting it -- the same thing test_block_order does for blocks."""
    model, x = make_model(), batch()
    visited = []
    handles = [site.register_forward_pre_hook(
                   lambda module, inputs, m=site: visited.append(m))
               for site, *_ in activation_sites(model)]
    try:
        model(x)
    finally:
        for handle in handles:
            handle.remove()
    assert visited == [site for site, *_ in activation_sites(model)]
    assert len(visited) == N_SITES


def test_sites_carry_their_stage_and_block():
    sites = activation_sites(make_model())
    assert sites[0][1:] == (0, -1)                  # the stem: stage 0, no block
    assert [stage for _, stage, _ in sites[1:]] == sum(
        ([stage] * 4 for stage in (1, 2, 3, 4)), [])
    assert [block for _, _, block in sites[1:]] == sum(
        ([b] * 2 for b in range(8)), [])


@pytest.mark.parametrize("which,count", [("all", 17), ("act1", 8), ("act2", 8)])
def test_which_selects_a_subset_of_the_sites(which, count):
    assert len(activation_sites(make_model(), which)) == count


@pytest.mark.parametrize("scope,groups", [("global", 1), ("stage", 5),
                                          ("block", 9), ("site", 17)])
def test_scope_sets_how_many_groups_there_are(scope, groups):
    sites = activation_sites(make_model())
    assert max(site_groups(sites, scope)) + 1 == groups


def test_scope_groups_are_contiguous_and_forward_ordered():
    sites = activation_sites(make_model())
    assigned = site_groups(sites, "stage")
    assert assigned == sorted(assigned)             # never revisits a group
    assert set(assigned) == set(range(5))


def test_reverse_flips_which_group_moves_first():
    """Bottom-up (input first) versus top-down (head first). It only means
    anything for a schedule that staggers groups, i.e. sequential."""
    sites = activation_sites(make_model())
    forward = site_groups(sites, "stage")
    backward = site_groups(sites, "stage", reverse=True)
    assert backward == [4 - g for g in forward]


def test_scope_with_a_subset_still_ranks_densely():
    """act1 only has no stem, so the stage indices start at 1 -- the group
    numbers must still be 0..3 with no hole, or the schedule vector and the
    groups disagree about how many there are."""
    sites = activation_sites(make_model(), "act1")
    assert set(site_groups(sites, "stage")) == set(range(4))


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def test_gate_writes_one_alpha_per_group_to_the_right_sites():
    model = make_model()
    gate = ActivationGate(model, scope="stage")
    gate.set([0.1, 0.2, 0.3, 0.4, 0.5])
    for site, stage, _ in activation_sites(model):
        assert site.alpha == pytest.approx(0.1 * (stage + 1))


def test_gate_broadcasts_a_scalar():
    model = make_model()
    ActivationGate(model, scope="site").set(0.25)
    assert all(site.alpha == 0.25 for site, *_ in activation_sites(model))


def test_gate_rejects_a_wrong_length_vector():
    gate = ActivationGate(make_model(), scope="stage")
    with pytest.raises(ValueError, match="5"):
        gate.set([0.1, 0.2])


def test_closing_the_gate_restores_the_relu_baseline():
    """Not "a leaky ReLU that happens to be small": exactly alpha=0, so the
    model is byte-for-byte the baseline again."""
    model, x = make_model(), batch()
    reference = model(x)
    gate = ActivationGate(model)
    gate.set(0.7)
    assert not torch.equal(model(x), reference)
    gate.close()
    assert torch.equal(model(x), reference)


def test_gate_at_restores_the_previous_alpha():
    model = make_model()
    gate = ActivationGate(model)
    gate.set(0.6)
    with gate.at(0.0):
        assert model.act0.alpha == 0.0
    assert model.act0.alpha == 0.6


def test_gate_leaves_a_subset_of_sites_alone():
    """a_sites=act2 must not touch act1, or the ablation is not an ablation."""
    model = make_model()
    ActivationGate(model, which="act2").set(0.5)
    assert model.layer1[0].act1.alpha == 0.0
    assert model.layer1[0].act2.alpha == 0.5


# --------------------------------------------------------------------------
# the policy: alpha via tau = 1 - alpha
# --------------------------------------------------------------------------

def alpha_cfg(**over):
    cfg = {"a_schedule": "linear", "a_start": 1.0, "a_end": 0.0,
           "a_ramp_start": 0.0, "a_ramp_end": 0.5, "a_stairs": 0}
    cfg.update(over)
    return cfg


@pytest.mark.parametrize("schedule", ["linear", "cosine", "sequential"])
def test_alpha_starts_at_a_start(schedule):
    assert alpha_at_cfg(0.0, 4, alpha_cfg(a_schedule=schedule)) == [1.0] * 4


@pytest.mark.parametrize("schedule", ["linear", "cosine", "staircase",
                                      "sequential"])
def test_alpha_reaches_a_end_by_ramp_end_and_stays(schedule):
    """The invariant that makes the accuracy comparable at all: a run whose
    alpha never reaches 0 trained a different model than the baseline."""
    cfg = alpha_cfg(a_schedule=schedule, a_stairs=4)
    for progress in (0.5, 0.7, 1.0):
        assert alpha_at_cfg(progress, 4, cfg) == pytest.approx([0.0] * 4)


@pytest.mark.parametrize("schedule", ["linear", "cosine", "staircase",
                                      "sequential"])
def test_alpha_is_monotone_decreasing_and_bounded(schedule):
    cfg = alpha_cfg(a_schedule=schedule, a_stairs=4)
    previous = [1.0] * 4
    for step in range(51):
        values = alpha_at_cfg(step / 50, 4, cfg)
        assert all(0.0 <= v <= 1.0 for v in values)
        assert all(v <= p + 1e-12 for v, p in zip(values, previous))
        previous = values


def test_a_start_below_one_never_starts_from_the_affine_network():
    """The arm that avoids alpha=1, where the Hessian is singular by
    construction and the implicit function theorem has nothing to say."""
    cfg = alpha_cfg(a_start=0.5)
    assert alpha_at_cfg(0.0, 1, cfg)[0] == pytest.approx(0.5)
    assert max(alpha_at_cfg(p / 20, 1, cfg)[0] for p in range(21)) <= 0.5


def test_const_holds_alpha_at_a_end_throughout():
    """The ablation arm: is a fixed LeakyReLU enough, with no homotopy at all?"""
    cfg = alpha_cfg(a_schedule="const", a_end=0.1)
    assert [alpha_at_cfg(p / 10, 1, cfg)[0] for p in range(11)] == \
        pytest.approx([0.1] * 11)


def test_two_stairs_is_the_jump_control():
    """The warm-start control needs no mechanism of its own: a staircase with
    two levels holds alpha at 1 and then drops it to 0, and ramp_end places
    the drop. Here the ramp spans [0, 0.6], so the jump lands at 0.3."""
    cfg = alpha_cfg(a_schedule="staircase", a_stairs=2, a_ramp_end=0.6)
    assert alpha_at_cfg(0.29, 1, cfg)[0] == pytest.approx(1.0)
    assert alpha_at_cfg(0.31, 1, cfg)[0] == pytest.approx(0.0)


def test_sequential_turns_groups_nonlinear_one_after_another():
    cfg = alpha_cfg(a_schedule="sequential")
    # a quarter of the way along a ramp that spans [0, 0.5]
    values = alpha_at_cfg(0.125, 4, cfg)
    assert values[0] == pytest.approx(0.0)          # already ReLU
    assert values[-1] == pytest.approx(1.0)         # still linear
    assert values == sorted(values)                 # nonlinear from the bottom up


def test_schedule_none_is_plain_relu():
    assert alpha_at_cfg(0.3, 3, alpha_cfg(a_schedule="none")) == [0.0] * 3


# --------------------------------------------------------------------------
# config: the keys, and what they refuse
# --------------------------------------------------------------------------

def load(*argv):
    return load_config(list(argv), default="baseline")


def test_the_baseline_config_leaves_the_activation_homotopy_off():
    """Leaving the a_* keys alone has to be byte-identical to before they
    existed, or every number already in runs/ stops being comparable."""
    cfg = load()
    assert cfg["a_schedule"] == "none"
    assert cfg["a_start"] == 1.0 and cfg["a_end"] == 0.0


@pytest.mark.parametrize("flags", [
    ("--a-schedule", "bogus"),
    ("--a-scope", "bogus"),
    ("--a-sites", "bogus"),
])
def test_an_unknown_choice_fails_at_launch(flags):
    with pytest.raises(SystemExit):
        load(*flags)


def test_alpha_must_decrease():
    with pytest.raises(SystemExit, match="a_end"):
        load("--a-schedule", "linear", "--a-start", "0.2", "--a-end", "0.8")


def test_the_ramp_must_leave_a_tail_at_a_end():
    """The same invariant s_ramp_end enforces: a run whose alpha never settles
    at its target trained a model its accuracy will never be compared to."""
    with pytest.raises(SystemExit, match="tail"):
        load("--a-schedule", "linear", "--a-ramp-end", "1.0")


def test_lr_restart_needs_a_staircase():
    with pytest.raises(SystemExit, match="staircase"):
        load("--a-schedule", "linear", "--a-lr-restart")


def test_reverse_needs_a_schedule_that_staggers_groups():
    """reverse under any other schedule silently does nothing, which is the
    kind of no-op that produces a run answering a question nobody asked."""
    with pytest.raises(SystemExit, match="sequential"):
        load("--a-schedule", "linear", "--a-scope", "stage", "--a-reverse")


def test_reverse_needs_more_than_one_group():
    with pytest.raises(SystemExit, match="scope"):
        load("--a-schedule", "sequential", "--a-scope", "global", "--a-reverse")


def test_a_staircase_needs_its_stairs():
    with pytest.raises(SystemExit, match="a_stairs"):
        load("--a-schedule", "staircase", "--a-stairs", "0")


def test_a_valid_activation_arm_resolves():
    cfg = load("--a-schedule", "linear", "--a-scope", "stage",
               "--a-ramp-end", "0.5")
    assert cfg["a_schedule"] == "linear" and cfg["a_scope"] == "stage"


# --------------------------------------------------------------------------
# continuation phases on the alpha axis
# --------------------------------------------------------------------------

def phase_cfg(**over):
    """A config with both axes present, since phase_bounds now reads both."""
    cfg = {"s_schedule": "const", "s_min": 0.0, "s_max": 1.0,
           "s_ramp_start": 0.0, "s_ramp_end": 0.5, "s_stairs": 0,
           "s_lr_restart": False,
           "a_schedule": "staircase", "a_start": 1.0, "a_end": 0.0,
           "a_ramp_start": 0.0, "a_ramp_end": 0.8, "a_stairs": 5,
           "a_lr_restart": True}
    cfg.update(over)
    return cfg


def test_no_alpha_phases_unless_asked_for():
    assert phase_bounds(phase_cfg(a_lr_restart=False)) == ()
    assert phase_bounds(phase_cfg(a_schedule="linear", a_lr_restart=False)) == ()


def test_an_alpha_phase_is_an_interval_of_constant_alpha():
    cfg = phase_cfg()
    bounds = phase_bounds(cfg)
    assert bounds[0] == 0.0 and bounds[-1] == 1.0
    assert bounds == pytest.approx((0.0, 0.16, 0.32, 0.48, 0.64, 1.0))
    values = [alpha_at_cfg((bounds[i] + bounds[i + 1]) / 2, 1, cfg)[0]
              for i in range(len(bounds) - 1)]
    assert values == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])
    for index in range(len(bounds) - 1):
        low, high = bounds[index], bounds[index + 1]
        inside = {round(alpha_at_cfg(low + (high - low) * f, 1, cfg)[0], 12)
                  for f in (0.01, 0.25, 0.5, 0.75, 0.99)}
        assert len(inside) == 1, f"alpha moves inside phase {index}"


def test_the_residual_axis_still_gets_its_own_phases():
    """Generalising phase_bounds must not have cost the s arms their phases."""
    cfg = phase_cfg(a_schedule="none", a_lr_restart=False,
                    s_schedule="staircase", s_stairs=5, s_lr_restart=True,
                    s_ramp_end=0.8)
    assert phase_bounds(cfg) == pytest.approx((0.0, 0.16, 0.32, 0.48, 0.64, 1.0))


def test_both_axes_at_once_give_the_union_of_their_boundaries():
    cfg = phase_cfg(a_stairs=2, a_ramp_end=0.8,
                    s_schedule="staircase", s_stairs=2, s_ramp_start=0.1,
                    s_ramp_end=0.5, s_lr_restart=True)
    # 0.3 is where s steps, 0.4 is where alpha does, and both are boundaries.
    assert phase_bounds(cfg) == pytest.approx((0.0, 0.3, 0.4, 1.0))


# --------------------------------------------------------------------------
# the per-block residual diagnostic has to see the real activation
# --------------------------------------------------------------------------

def test_residual_ratio_uses_the_block_activation_not_a_hardcoded_relu():
    """residual_ratio recomputes F(h) by hand. If it calls F.relu directly it
    measures a network that is not running, and every ratio reported during an
    activation-homotopy run is wrong while looking entirely plausible."""
    model, x = make_model(), batch()
    at_relu = residual_ratio(model, x)["blocks"][0]["f_rms"]
    # act1 only, so act0 stays at ReLU and block 0 is handed the identical
    # input. f_rms can then move for exactly one reason: the branch's own
    # activation. Gating every site instead would change h too, and the test
    # would pass whether or not the bug was fixed.
    ActivationGate(model, which="act1").set(1.0)
    at_linear = residual_ratio(model, x)["blocks"][0]["f_rms"]
    assert at_relu != pytest.approx(at_linear, rel=1e-3)


# --------------------------------------------------------------------------
# the diagnostic that decides whether a run is honest
# --------------------------------------------------------------------------

def make_split(n=64, seed=3):
    generator = torch.Generator().manual_seed(seed)
    return Split(torch.randint(0, 256, (n, 3, 32, 32), generator=generator,
                               dtype=torch.uint8),
                 torch.randint(0, 10, (n,), generator=generator),
                 torch.device("cpu"))


def test_activation_stats_reports_one_row_per_site():
    rows = activation_stats(make_model(), batch())["sites"]
    assert len(rows) == N_SITES
    assert set(rows[0]) >= {"site", "stage", "block", "alpha", "rms",
                            "neg_frac", "linear_gap", "dead_frac"}


def test_neg_frac_survives_the_inplace_relu():
    """The statistics have to be taken before phi_alpha runs, not after.

    Activation is inplace by default, so a hook that stashes the tensor and
    measures it later reads a tensor whose negatives have already been erased.
    neg_frac would then be 0 everywhere and the shift escape below would be
    undetectable -- silently, and only under the default inplace=True.
    """
    rows = activation_stats(make_model(), batch())["sites"]
    assert rows[0]["neg_frac"] > 0.1


def test_linear_gap_is_zero_when_the_network_is_linear():
    """alpha=1 means phi_alpha is the identity, so the gap from linear is 0."""
    model = make_model()
    ActivationGate(model).set(1.0)
    rows = activation_stats(model, batch())["sites"]
    assert max(row["linear_gap"] for row in rows) == pytest.approx(0.0)


def test_linear_gap_matches_its_closed_form():
    """linear_gap = rms(phi_alpha(h) - h)/rms(h) = (1-alpha)*rms(relu(-h))/rms(h)."""
    model, x = make_model(), batch()
    ActivationGate(model).set(0.3)
    row = activation_stats(model, x)["sites"][0]
    with torch.no_grad():
        h = model.bn1(model.conv1(x))
    expected = 0.7 * float(F.relu(-h).pow(2).mean().sqrt()
                           / h.pow(2).mean().sqrt())
    assert row["linear_gap"] == pytest.approx(expected, rel=1e-4)


def test_linear_gap_catches_the_shift_escape():
    """The escape alpha leaves open: alpha cannot be scaled away, but BN can
    push beta positive until every pre-activation is positive, and phi_alpha is
    then the identity no matter what alpha says. A run whose linear_gap stays
    near zero while alpha falls is a baseline wearing a costume, and this is
    the number that says so."""
    model, x = make_model(), batch()
    honest = activation_stats(model, x)["sites"][0]["linear_gap"]
    with torch.no_grad():
        model.bn1.bias.fill_(50.0)          # every pre-activation now positive
    escaped = activation_stats(model, x)["sites"][0]
    assert honest > 0.3
    assert escaped["linear_gap"] < 0.01
    assert escaped["neg_frac"] == pytest.approx(0.0)


def test_activation_stats_leaves_the_alphas_alone():
    model = make_model()
    ActivationGate(model).set(0.4)
    activation_stats(model, batch())
    assert model.act0.alpha == 0.4


def test_loss_vs_alpha_agrees_with_a_direct_evaluation():
    model, split = make_model(), make_split()
    gate = ActivationGate(model)
    swept = loss_vs_alpha(model, split, [0.0, 0.5, 1.0], gate, batch_size=32)
    for index, alpha in enumerate([0.0, 0.5, 1.0]):
        gate.set(alpha)
        direct, _ = evaluate_at(model, split, batch_size=32)
        assert swept["loss"][index] == pytest.approx(direct, rel=1e-5)
    assert gate.get() == [1.0]              # restored to what it held


def test_functional_distance_is_zero_between_a_point_and_itself():
    model, split = make_model(), make_split()
    weights = get_weights(model)
    result = functional_distance(model, weights, weights, split, batch_size=32)
    assert result["disagreement"] == 0.0
    assert result["sym_kl"] == pytest.approx(0.0, abs=1e-6)


def test_functional_distance_grows_with_the_gap_between_two_solutions():
    """The measure the continuation needs: two points on a branch can be far
    apart in parameter space and compute nearly the same function -- BN scale
    invariance alone guarantees it -- so ||theta_a - theta_b|| is the wrong
    continuity signal and this is the right one."""
    model, split = make_model(), make_split()
    origin = get_weights(model)
    direction = random_direction(model, seed=0)
    near = functional_distance(model, origin, origin + 0.05 * direction, split,
                               batch_size=32)
    far = functional_distance(model, origin, origin + 2.0 * direction, split,
                              batch_size=32)
    assert near["sym_kl"] < far["sym_kl"]


# --------------------------------------------------------------------------
# the alpha=0 readout, and the BatchNorm trap under it
# --------------------------------------------------------------------------

def test_the_readout_leaves_batchnorm_exactly_as_it_found_it():
    """It re-estimates the running statistics to measure, then must put the
    training run's own statistics back. Without the restore, every diagnostic
    epoch would silently hand the next epoch a different model."""
    model, split = make_model(), make_split()
    model.train()
    with torch.no_grad():                   # give the buffers something to lose
        model.bn1.running_mean.fill_(0.3)
        model.bn1.running_var.fill_(2.0)
    gate = ActivationGate(model)
    gate.set(0.8)
    readout_at_alpha_zero(model, gate, split, split, {"eval_batch_size": 32,
                                                      "a_bn_batches": 2})
    assert model.bn1.running_mean.allclose(torch.full_like(model.bn1.running_mean, 0.3))
    assert model.bn1.running_var.allclose(torch.full_like(model.bn1.running_var, 2.0))
    assert gate.get() == [0.8]
    assert model.training


def test_the_readout_evaluates_through_re_estimated_statistics():
    """The trap this exists for: dropping alpha to 0 removes the whole negative
    mass of every activation, so statistics gathered at alpha>0 describe a
    distribution the network no longer produces. Reading through them makes a
    working homotopy look like one that does not transfer.

    Checked on the statistics the evaluation actually sees rather than on the
    accuracy it returns: on an untrained network over random labels both
    readings are chance, so accuracy cannot tell the two apart even though the
    models differ.
    """
    model, split = make_model(), make_split()
    model.train()
    with torch.no_grad():
        model.bn1.running_mean.fill_(9.0)   # statistics no network produces
    gate = ActivationGate(model)
    gate.set(1.0)

    seen = []
    handle = model.bn1.register_forward_hook(
        lambda *_: seen.append(float(model.bn1.running_mean.mean())))
    base = {"eval_batch_size": 32}
    try:
        readout_at_alpha_zero(model, gate, split, split, dict(base, a_bn_batches=2))
        during_fresh = seen[-1]
        seen.clear()
        readout_at_alpha_zero(model, gate, split, split, dict(base, a_bn_batches=0))
        during_stale = seen[-1]
    finally:
        handle.remove()

    assert during_stale == pytest.approx(9.0)   # a_bn_batches=0 is the stale read
    assert abs(during_fresh) < 1.0              # re-estimated from real activations


def test_a_config_with_no_activation_homotopy_builds_no_gate():
    assert build_activation_gate(make_model(), {"a_schedule": "none"}) is None


def test_functional_distance_compares_each_endpoint_at_its_own_alpha():
    """Branch continuity is about the networks theta*(alpha_k) defines, and
    each of those is read at the alpha it was trained at. The same weights at
    two different alphas are two different functions, and the measure has to
    say so."""
    model, split = make_model(), make_split()
    weights = get_weights(model)
    gate = ActivationGate(model)
    same = functional_distance(model, weights, weights, split, alpha=0.0,
                               gate=gate, batch_size=32)
    across = functional_distance(model, weights, weights, split,
                                 alpha=(1.0, 0.0), gate=gate, batch_size=32)
    assert same["sym_kl"] == pytest.approx(0.0, abs=1e-6)
    assert across["sym_kl"] > 1e-4
    assert across["weight_distance"] == 0.0     # the weights never moved
    assert gate.get() == [0.0]


def test_the_readout_reports_loss_as_well_as_accuracy():
    """test_acc_at_a0 has no loss counterpart, so the only test loss on record
    is measured at the arm's current alpha -- which during the ramp is the loss
    of a network that is not the ResNet, and is no more comparable across arms
    than train_loss is. Both numbers come from the same forward pass, so there
    is no reason to throw one away."""
    model, split = make_model(), make_split()
    model.train()
    gate = ActivationGate(model)
    gate.set(0.8)
    out = readout_at_alpha_zero(model, gate, split, split,
                                {"eval_batch_size": 32, "a_bn_batches": 2})
    assert set(out) == {"acc", "loss"}
    assert 0.0 <= out["acc"] <= 1.0
    assert out["loss"] > 0.0


def test_a_staircase_may_span_the_whole_run():
    """The tail rule exists so a run always ends on the model its accuracy is
    compared against. A staircase's last plateau *is* a_end -- progress past
    ramp_end returns the same level -- so the guarantee is structural and the
    ramp may cover the whole run. That is what lets five plateaus be exactly
    ten epochs each in a fifty-epoch budget, instead of four short ones and a
    long tail."""
    cfg = load_config(["--a-schedule", "staircase", "--a-stairs", "5",
                       "--a-ramp-end", "1.0"], default="baseline")
    assert cfg["a_ramp_end"] == 1.0
    levels = [alpha_at_cfg(e / 50, 1, cfg)[0] for e in range(50)]
    assert sorted(set(levels), reverse=True) == pytest.approx([1.0, .75, .5, .25, 0.0])
    assert levels.count(0.0) == 10                  # ten epochs of real ReLU
    assert levels[-1] == 0.0


def test_a_continuous_ramp_still_may_not_span_the_whole_run():
    """The relaxation is for staircase only: under a linear ramp, ramp_end=1
    means alpha reaches 0 on the very last epoch and the target problem is
    never optimised at all."""
    with pytest.raises(SystemExit, match="tail"):
        load_config(["--a-schedule", "linear", "--a-ramp-end", "1.0"],
                    default="baseline")

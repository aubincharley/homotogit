"""Adaptive predictor-corrector continuation.

Two invariants carry the whole method and are tested adversarially rather than
happily: sigma must **never** increase whatever the measured sensitivities say,
and the run must reach exactly zero.  Both are what separate a homotopy from a
learned blur layer, and a free adaptive rule gives 19 independent chances to
break either one.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from continuation.adaptive import (SIGMA_FLOOR, AdaptiveSiteController,
                                   GradNormTracker, SensitivityStepper,
                                   phi_of_sigma)
from continuation.campaign_ops import N_SITES, SiteController, attach_sites
from continuation.config import ModelConfig
from continuation.models import build_model
from continuation.transforms.gaussian import (GaussianSmoothing,
                                              blur_with_sigma_grad,
                                              gaussian_kernel_1d_tensor)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


# --------------------------------------------------------------------------
# the differentiable sigma path
# --------------------------------------------------------------------------

def test_gradcheck_of_dL_dsigma():
    """float64, tiny tensor, strictly interior sigma -- the repo's gradcheck pattern."""
    x = torch.rand(1, 1, 8, 8, generator=gen(15), dtype=torch.float64)
    sigma = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda s: blur_with_sigma_grad(x, s, 4).square().sum(), (sigma,),
        eps=1e-6, atol=1e-6)


def test_the_sigma_gradient_is_actually_nonzero():
    """Guards against a gradcheck that passes because everything is zero."""
    x = torch.rand(1, 3, 16, 16, generator=gen(3), dtype=torch.float64)
    sigma = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    g, = torch.autograd.grad(blur_with_sigma_grad(x, sigma, 4).square().sum(), sigma)
    assert torch.isfinite(g) and abs(float(g)) > 1e-6


def test_renormalisation_forces_the_tap_derivative_to_sum_to_zero():
    """sum(k) == 1 identically in sigma, so sum(dk/dsigma) == 0 exactly."""
    for s0 in (0.2, 0.5, 0.9):
        sigma = torch.tensor(s0, dtype=torch.float64, requires_grad=True)
        k = gaussian_kernel_1d_tensor(sigma, 4)
        assert abs(float(k.sum().detach()) - 1.0) < 1e-12
        g, = torch.autograd.grad(k.sum(), sigma)
        assert abs(float(g)) < 1e-12


def test_tensor_path_is_bitwise_identical_to_the_float_hot_path():
    """The probe must measure the operator that training actually uses."""
    x = torch.randn(2, 3, 16, 16, generator=gen(4), dtype=torch.float64)
    ref = GaussianSmoothing(sigma_max=1.0, truncate=4.0)
    for s0 in (0.15, 0.4, 1.0):
        a = blur_with_sigma_grad(x, torch.tensor(s0, dtype=torch.float64), ref.radius)
        assert torch.equal(a, ref(x, s0)), "tensor path diverges at sigma=%g" % s0


def test_tensor_path_rejects_a_float_sigma():
    x = torch.randn(1, 1, 8, 8)
    with pytest.raises(TypeError, match="tensor sigma"):
        blur_with_sigma_grad(x, 0.5, 4)


def test_small_map_uses_the_explicit_reflection_and_stays_differentiable():
    """radius 4 on a 4x4 stage-3 map -- the case native reflect padding refuses."""
    x = torch.rand(1, 2, 4, 4, generator=gen(6), dtype=torch.float64)
    sigma = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    g, = torch.autograd.grad(blur_with_sigma_grad(x, sigma, 4).square().sum(), sigma)
    assert torch.isfinite(g)


# --------------------------------------------------------------------------
# corrector-converged trigger
# --------------------------------------------------------------------------

def test_tracker_waits_for_min_steps_even_on_a_flat_signal():
    t = GradNormTracker(min_steps=10, patience=1, tol=0.0)
    for _ in range(9):
        t.update(5.0)
        assert not t.should_step()
    t.update(5.0)
    assert t.should_step()


def test_tracker_does_not_fire_while_the_norm_keeps_improving():
    t = GradNormTracker(beta=0.5, min_steps=2, patience=3, tol=0.01)
    g = 10.0
    for _ in range(40):
        g *= 0.8
        t.update(g)
    assert not t.should_step()


def test_tracker_fires_on_a_plateau():
    t = GradNormTracker(beta=0.5, min_steps=2, patience=3, tol=0.01)
    for _ in range(30):
        t.update(1.0)
    assert t.should_step()


def test_tracker_max_steps_forces_a_step():
    t = GradNormTracker(beta=0.5, min_steps=2, patience=10 ** 6, max_steps=5)
    for _ in range(5):
        t.update(1.0)
    assert t.should_step()


def test_tracker_reset_clears_the_stage():
    t = GradNormTracker(min_steps=1, patience=1, tol=0.0)
    for _ in range(5):
        t.update(1.0)
    assert t.should_step()
    t.reset_stage()
    assert t.steps == 0 and t.ema is None and not t.should_step()


# --------------------------------------------------------------------------
# the step rule: monotonicity is a contract
# --------------------------------------------------------------------------

def test_sigma_never_increases_whatever_the_gradient_sign():
    """Adversarial: the loss-descent direction may point AWAY from the target."""
    st = SensitivityStepper(dmin=0.01, dmax=0.3)
    sites = tuple(range(N_SITES))
    row = [0.9] * N_SITES
    rng = torch.Generator().manual_seed(11)
    stages = 9
    for k in range(stages):
        g = (torch.randn(N_SITES, generator=rng) * 10).tolist()   # both signs
        nxt = st.step(row, g, sites, stages_left=stages - k)
        assert all(nxt[l] <= row[l] + 1e-12 for l in sites), "sigma rose"
        assert all(v >= 0.0 and v == v for v in nxt)
        row = nxt
    assert all(v == 0.0 for v in row), "did not reach the target endpoint"


def test_a_gradient_spanning_16_orders_cannot_produce_a_bang_bang_front():
    """The failure mode kappa exists to prevent.

    Dividing by a sensitivity is positive feedback: a site that anneals faster
    gets a smaller gradient, hence a bigger step.  Unbounded, sites collapse one
    at a time.  kappa caps the per-stage ratio, so the spread stays bounded.
    """
    st = SensitivityStepper(kappa=3.0)
    sites = tuple(range(N_SITES))
    row = [0.9] * N_SITES
    g = [1.0] * N_SITES
    g[13], g[0] = 1e8, 1e-8
    for k in range(5):
        row = st.step(row, g, sites, stages_left=6 - k)
        live = [v for v in row if v > 0]
        if len(live) > 1:
            assert max(live) / min(live) <= st.kappa ** len(live), "front ran away"
    assert all(0.0 <= v <= 0.9 for v in row)


def test_non_finite_sensitivities_fall_back_to_the_schedule_and_are_recorded():
    st = SensitivityStepper()
    sites = tuple(range(N_SITES))
    g = [1.0] * N_SITES
    g[4], g[9] = float("nan"), float("inf")
    out = st.step([0.9] * N_SITES, g, sites, stages_left=6)
    assert all(v == 0.0 or 0.0 < v < 0.9 for v in out)
    assert st.probe_invalid == 1, "an unusable probe must be counted, not hidden"


def test_equal_sensitivities_reproduce_a_uniform_step():
    st = SensitivityStepper(delta_ref=0.1, dmin=0.01, dmax=0.3)
    sites = tuple(range(N_SITES))
    out = st.step([0.9] * N_SITES, [3.3] * N_SITES, sites)
    assert len(set(out)) == 1


def test_a_more_sensitive_site_takes_a_smaller_step():
    st = SensitivityStepper(dmin=0.001, dmax=10.0, kappa=100.0)
    sites = tuple(range(N_SITES))
    g = [1.0] * N_SITES
    g[13] = 100.0                       # stage-3 site is very sensitive
    g[0] = 0.01                         # stem is insensitive
    out = st.step([0.9] * N_SITES, g, sites, stages_left=6)
    assert (0.9 - out[13]) < (0.9 - out[1]) < (0.9 - out[0])


def test_dmin_below_the_table_quantum_is_rejected():
    """Otherwise a step rounds to no change and the run stalls forever."""
    with pytest.raises(ValueError, match="table quantum"):
        SensitivityStepper(dmin=1e-7)


def test_kappa_below_one_is_rejected():
    with pytest.raises(ValueError, match="kappa"):
        SensitivityStepper(kappa=0.5)


# --------------------------------------------------------------------------
# the phi reparameterisation
# --------------------------------------------------------------------------

def test_phi_is_monotone_and_hits_the_documented_values():
    assert phi_of_sigma(0.0) == 0.0
    for a, b in zip([0.12, 0.25, 0.5, 0.7, 1.0], [0.25, 0.5, 0.7, 1.0, 1.5]):
        assert phi_of_sigma(a) < phi_of_sigma(b)
    assert abs(phi_of_sigma(1.0) - 0.6065) < 1e-3
    assert phi_of_sigma(0.12) < 1e-14      # numerically the identity


def test_phi_is_far_better_conditioned_than_sigma_for_this_operator():
    """The measured justification for the reparameterisation.

    Not heat time: a = sigma^2/2 is the continuum answer and is wrong for a
    kernel truncated at radius 4 on a unit grid.
    """
    x = torch.randn(1, 4, 16, 16, generator=gen(0), dtype=torch.float64)
    w = torch.randn(1, 4, 16, 16, generator=gen(1), dtype=torch.float64)
    gs, gp = [], []
    for s0 in (1.0, 0.7, 0.5, 0.3, 0.25, 0.2, 0.15):
        s = torch.tensor(s0, dtype=torch.float64, requires_grad=True)
        g, = torch.autograd.grad((blur_with_sigma_grad(x, s, 4) * w).sum(), s)
        gs.append(abs(float(g)))
        gp.append(abs(float(g)) / (phi_of_sigma(s0) / s0 ** 3))
    cond_sigma = max(gs) / min(gs)
    cond_phi = max(gp) / min(gp)
    assert cond_sigma > 1e6, "expected sigma-space to be badly conditioned"
    assert cond_phi < 1e3, "phi-space should be well conditioned"
    assert cond_phi < cond_sigma / 1e3


def test_step_snaps_to_exact_zero_on_the_final_stage():
    st = SensitivityStepper(dmin=0.05, dmax=0.05, sigma_floor=SIGMA_FLOOR)
    sites = tuple(range(N_SITES))
    # not on the way there: the site parks at the floor to stay exposure-matched
    held = st.step([0.13] * N_SITES, [1.0] * N_SITES, sites, stages_left=5)
    assert all(v == SIGMA_FLOOR for v in held)
    # but the final stage lands on the exact target
    out = st.step(held, [1.0] * N_SITES, sites, stages_left=1)
    assert all(v == 0.0 for v in out)


def test_parking_at_the_floor_never_raises_sigma():
    """A site already below the floor must not be pushed back up to it."""
    st = SensitivityStepper(sigma_floor=SIGMA_FLOOR)
    row = [0.05] * N_SITES
    out = st.step(row, [1.0] * N_SITES, tuple(range(N_SITES)), stages_left=5)
    assert all(out[l] <= row[l] for l in range(N_SITES))


def test_zero_sites_are_left_alone():
    st = SensitivityStepper()
    row = [0.0] * N_SITES
    assert st.step(row, [1.0] * N_SITES, tuple(range(N_SITES))) == row


def test_ramp_reaches_exactly_zero_within_the_remaining_stages():
    st = SensitivityStepper(sigma_floor=SIGMA_FLOOR)
    sites = tuple(range(N_SITES))
    row = [0.85] * N_SITES
    for left in range(4, 0, -1):
        row = st.ramp(row, left, sites)
    assert all(v == 0.0 for v in row)


# --------------------------------------------------------------------------
# the controller
# --------------------------------------------------------------------------

def make_model(seed=5):
    return build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=seed)


class IdPipe:
    def __call__(self, x, eta, res=None):
        return x


def test_sigma_init_is_validated_against_sigma_max():
    with pytest.raises(ValueError, match="outside"):
        AdaptiveSiteController([2.0] * N_SITES)
    with pytest.raises(ValueError, match="need %d" % N_SITES):
        AdaptiveSiteController([0.5] * 3)


def test_measure_returns_one_gradient_per_filtered_site():
    torch.manual_seed(0)
    model, pipe = make_model(), IdPipe()
    row = [0.6] * N_SITES
    row[3] = 0.0                                    # a bypassed site
    ctrl = AdaptiveSiteController(row)
    handles = attach_sites(model, ctrl)
    try:
        g = ctrl.measure(model, pipe, torch.randn(8, 3, 32, 32),
                         torch.randint(0, 10, (8,)))
    finally:
        for h in handles:
            h.remove()
    assert len(g) == N_SITES
    assert g[3] == 0.0, "a bypassed site must report no sensitivity"
    assert sum(1 for v in g if v != 0.0) > 10, "most sites should be sensitive"
    assert all(torch.isfinite(torch.tensor(v)) for v in g)


def test_measure_leaves_bn_buffers_mode_and_grads_untouched():
    torch.manual_seed(1)
    model, pipe = make_model(), IdPipe()
    model.train()
    bn_before = [b.clone() for b in model.buffers()]
    ctrl = AdaptiveSiteController([0.6] * N_SITES)
    handles = attach_sites(model, ctrl)
    try:
        ctrl.measure(model, pipe, torch.randn(8, 3, 32, 32),
                     torch.randint(0, 10, (8,)))
    finally:
        for h in handles:
            h.remove()
    assert model.training, "prior mode must be restored"
    for a, b in zip(bn_before, model.buffers()):
        assert torch.equal(a, b), "BatchNorm statistics were disturbed"
    assert all(p.grad is None for p in model.parameters()), "model .grad polluted"


def test_controller_terminates_under_randomised_triggers():
    rng = torch.Generator().manual_seed(21)
    ctrl = AdaptiveSiteController([1.0] * N_SITES,
                                  stepper=SensitivityStepper(delta_ref=0.05,
                                                             dmin=0.01, dmax=0.2))
    epochs, deadline = 40, 30
    for e in range(epochs):
        ctrl.set_epoch(e)
        if ctrl.all_zero():
            continue
        g = (torch.randn(N_SITES, generator=rng) * 50).tolist()
        ctrl.advance(g, stages_left=max(epochs - e, 1), force_ramp=e >= deadline)
    assert ctrl.all_zero(), "never reached the target objective"
    assert len(ctrl.realised) == epochs


def test_deadline_ramp_is_recorded_not_silent():
    ctrl = AdaptiveSiteController([0.9] * N_SITES)
    assert not ctrl.deadline_fired
    ctrl.advance([1.0] * N_SITES, stages_left=2, force_ramp=True)
    assert ctrl.deadline_fired
    assert ctrl.describe()["deadline_fired"] is True


def test_snapping_to_zero_is_recorded_per_site():
    ctrl = AdaptiveSiteController([0.13] * N_SITES,
                                  stepper=SensitivityStepper(dmin=0.05, dmax=0.05))
    ctrl.advance([1.0] * N_SITES, stages_left=1)
    assert all(v == 1 for v in ctrl.describe()["snapped_at_step"])


# --------------------------------------------------------------------------
# an adaptive run must be replayable as a fixed schedule
# --------------------------------------------------------------------------

def test_realised_table_replays_bitwise_through_the_fixed_controller():
    """Without this an adaptive run is not comparable to anything."""
    torch.manual_seed(2)
    rng = torch.Generator().manual_seed(7)
    x = torch.randn(2, 3, 32, 32)

    adaptive = AdaptiveSiteController([0.9] * N_SITES,
                                      stepper=SensitivityStepper(delta_ref=0.08,
                                                                 dmin=0.02, dmax=0.25))
    model = make_model()
    handles = attach_sites(model, adaptive)
    outs = []
    try:
        for e in range(6):
            adaptive.set_epoch(e)
            model.eval()
            outs.append(model(x).detach().clone())
            if not adaptive.all_zero():
                adaptive.advance((torch.randn(N_SITES, generator=rng) * 20).tolist())
    finally:
        for h in handles:
            h.remove()

    replay = SiteController("gaussian", sigma_table=adaptive.realised)
    model2 = make_model()
    handles = attach_sites(model2, replay)
    try:
        for e in range(6):
            replay.set_epoch(e)
            model2.eval()
            assert torch.equal(model2(x), outs[e]), "replay diverged at epoch %d" % e
    finally:
        for h in handles:
            h.remove()


def test_realised_table_carries_the_usual_provenance():
    ctrl = AdaptiveSiteController([0.5] * N_SITES)
    for e in range(3):
        ctrl.set_epoch(e)
        ctrl.advance([1.0] * N_SITES)
    d = ctrl.describe()
    assert len(d["realised_table"]) == 3
    assert d["controller"] == "adaptive_predictor_corrector"
    assert d["predictor"].startswith("warm start")
    assert d["stepper"]["normalisation"].startswith("global median")
    # and it is a valid input to the fixed-table constructor
    SiteController("gaussian", sigma_table=ctrl.realised)


def test_no_site_reaches_zero_early_so_arms_stay_exposure_matched():
    """The curriculum-vs-plain confound, closed by construction.

    An adaptive arm that hit sigma=0 early would simply train longer on the
    target objective than its fixed controls, and that alone could produce an
    accuracy difference.  A sprinting site must park at the floor instead.
    """
    st = SensitivityStepper(kappa=3.0)
    sites = tuple(range(N_SITES))
    row = [1.0] * N_SITES
    g = [1.0] * N_SITES
    g[13] = 1e-12                       # wildly insensitive: wants to sprint
    for k in range(9):
        row = st.step(row, g, sites, stages_left=9 - k)
        if k < 8:
            assert all(v > 0.0 for v in row), "site reached zero at stage %d" % k
            assert row[13] >= st.sigma_floor
    assert all(v == 0.0 for v in row), "all sites must arrive together"


def test_allow_early_zero_opts_out_of_exposure_matching():
    st = SensitivityStepper(kappa=3.0, allow_early_zero=True)
    row = st.step([0.13] * N_SITES, [1.0] * N_SITES, tuple(range(N_SITES)),
                  stages_left=9)
    assert all(v == 0.0 for v in row)


# --------------------------------------------------------------------------
# the shipped defaults must actually let the controller adapt
# --------------------------------------------------------------------------

def _study_defaults():
    """The argparse defaults the study script actually ships."""
    import importlib
    m = importlib.import_module("scripts.study_per_layer_cpu")
    import argparse
    saved = argparse.ArgumentParser.parse_args
    captured = {}

    def grab(self, argv=None):
        captured.update({a.dest: a.default for a in self._actions})
        raise SystemExit(0)
    argparse.ArgumentParser.parse_args = grab
    try:
        m.main([])
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = saved
    return captured


def test_shipped_defaults_do_not_clip_every_site_to_dmin():
    """The regression test for a real silent failure.

    A stale ``delta_ref`` (0.15, left over from the pre-phi loss-budget
    parameterisation) made every achievable step fall below ``dmin``, so all 19
    sites clipped to the same value and a full GPU run emitted a perfectly
    uniform linear ramp while reporting itself as adaptive.  The probe was fine;
    the step rule discarded its output.  Assert the defaults leave the whole
    kappa range expressible.
    """
    d = _study_defaults()
    st = SensitivityStepper(delta_ref=d["delta_ref"], dmin=d["dmin"],
                            dmax=d["dmax"], kappa=3.0)
    row, stages_left = [1.0] * N_SITES, 21
    base = row[0] / stages_left
    lo = base * st.delta_ref / st.kappa
    hi = base * st.delta_ref * st.kappa
    assert lo > st.dmin, ("dmin=%g swallows the low end of the kappa range (%g): "
                          "every site clips to dmin and the schedule goes uniform"
                          % (st.dmin, lo))
    assert hi < st.dmax, "dmax truncates the high end of the kappa range"


def test_shipped_defaults_produce_a_non_uniform_table():
    """End-to-end: realistic per-site sensitivities must survive into the table."""
    d = _study_defaults()
    st = SensitivityStepper(delta_ref=d["delta_ref"], dmin=d["dmin"],
                            dmax=d["dmax"], kappa=3.0)
    rng = torch.Generator().manual_seed(4)
    row = [1.0] * N_SITES
    # spread comparable to what the GPU probe actually measured (30-150x)
    g = (torch.rand(N_SITES, generator=rng) * 0.3 + 0.002).tolist()
    for k in range(6):
        row = st.step(row, g, tuple(range(N_SITES)), stages_left=21 - k)
    assert max(row) - min(row) > 0.05, ("sites did not differentiate: spread %.4f"
                                        % (max(row) - min(row)))

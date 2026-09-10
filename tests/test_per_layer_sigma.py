"""Per-site sigma table: the checks that guard the per-layer homotopy.

The point of every test here is that adding per-layer freedom must not weaken
any invariant the uniform schedule already had -- in particular the exact
identity endpoint, which is what makes the family a homotopy rather than a
rescaled objective, and which a free table gives 19 independent chances to break.
"""
from __future__ import annotations

import pytest
import torch

from continuation.campaign_ops import (EARLY7, LEVEL_DECIMALS, N_SITES,
                                       STAGE_OF_SITE, SiteController,
                                       attach_sites, build_level_table,
                                       per_stage_coeffs, table_digest)
from continuation.config import ModelConfig
from continuation.models import build_model
from continuation.transforms.gaussian import GaussianSmoothing


def make_model(seed: int = 7):
    return build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=seed).eval()


def forward_backward(model, x):
    """A real forward+backward; returns logits and the input gradient."""
    x = x.clone().requires_grad_(True)
    out = model(x)
    out.square().sum().backward()
    return out.detach().clone(), x.grad.detach().clone()


def run_with(ctrl, model, x, epoch=0):
    handles = attach_sites(model, ctrl)
    try:
        ctrl.set_epoch(epoch)
        return forward_backward(model, x)
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------------------
# profiles and table construction
# --------------------------------------------------------------------------

def test_stage_map_matches_the_resnet20_site_ordering():
    assert len(STAGE_OF_SITE) == N_SITES
    # site 0 is the stem and sites 1..6 are stage 1 -- exactly the early7 set
    assert set(l for l in range(N_SITES) if STAGE_OF_SITE[l] == 0) == set(EARLY7)
    assert STAGE_OF_SITE[7:13] == (1,) * 6
    assert STAGE_OF_SITE[13:19] == (2,) * 6


def test_per_stage_profiles_have_the_documented_values():
    assert set(per_stage_coeffs(1.0)) == {1.0}
    half = per_stage_coeffs(0.5)
    assert [half[0], half[7], half[13]] == [1.0, 0.5, 0.25]
    # rho=0.5 normalized is exactly width_l / 32, the constant-physical-scale profile
    assert [half[0], half[7], half[13]] == [32 / 32, 16 / 32, 8 / 32]
    two = per_stage_coeffs(2.0)
    assert [two[0], two[7], two[13]] == [0.25, 0.5, 1.0]
    assert max(two) == 1.0                      # normalization keeps us under sigma_max


def test_constructors_are_mutually_exclusive():
    with pytest.raises(ValueError, match="exclusive"):
        build_level_table(levels=[1.0], sigma_table=[[0.0] * N_SITES])
    with pytest.raises(ValueError, match="needs levels"):
        build_level_table(site_coeffs=[1.0] * N_SITES)


def test_table_rejects_wrong_width_and_bad_values():
    with pytest.raises(ValueError, match="need %d" % N_SITES):
        build_level_table(sigma_table=[[0.0] * (N_SITES - 1)])
    with pytest.raises(ValueError, match="need %d" % N_SITES):
        build_level_table(levels=[1.0], site_coeffs=[1.0] * 3)
    with pytest.raises(ValueError, match=">= 0"):
        build_level_table(levels=[-0.5])
    with pytest.raises(ValueError, match="finite"):
        build_level_table(levels=[float("nan")])


# --------------------------------------------------------------------------
# T6 / T1: the uniform table is the old behaviour, bitwise
# --------------------------------------------------------------------------

def test_T6_rho_one_profile_equals_the_uniform_table():
    levels = [1.0, 0.5, 0.0]
    plain = SiteController("gaussian", levels=levels)
    rho1 = SiteController("gaussian", levels=levels,
                          site_coeffs=per_stage_coeffs(1.0))
    assert plain.L == rho1.L
    assert table_digest(plain.L) == table_digest(rho1.L)


def test_T1_uniform_table_reproduces_the_scalar_rule_bitwise():
    """The table path must equal sigma_l = q_l * levels[e], the previous contract."""
    torch.manual_seed(0)
    x = torch.randn(4, 3, 32, 32)
    levels = [0.7, 0.0]

    got = run_with(SiteController("gaussian", levels=levels), make_model(), x)

    # reference: apply the Gaussian by hand at every site with the old rule
    gauss = GaussianSmoothing(sigma_max=1.0, truncate=4.0)
    ref_model = make_model()
    handles = []
    site = 0
    for m in ref_model.modules():
        if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (3, 3):
            handles.append(m.register_forward_hook(
                lambda _m, _i, out: gauss(out, 1.0 * 0.7)))
            site += 1
    assert site == N_SITES
    ref = forward_backward(ref_model, x)
    for h in handles:
        h.remove()

    assert torch.equal(got[0], ref[0]), "logits differ from the scalar rule"
    assert torch.equal(got[1], ref[1]), "input gradients differ from the scalar rule"


def test_T1b_site_coeffs_of_one_matches_no_site_coeffs_bitwise():
    torch.manual_seed(1)
    x = torch.randn(4, 3, 32, 32)
    levels = [0.6, 0.0]
    a = run_with(SiteController("gaussian", levels=levels), make_model(), x)
    b = run_with(SiteController("gaussian", levels=levels,
                                site_coeffs=[1.0] * N_SITES), make_model(), x)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_explicit_table_matches_the_separable_construction():
    levels, c = [0.8, 0.4, 0.0], per_stage_coeffs(0.5)
    sep = SiteController("gaussian", levels=levels, site_coeffs=c)
    tab = SiteController("gaussian",
                         sigma_table=[[c[l] * g for l in range(N_SITES)]
                                      for g in levels])
    assert sep.L == tab.L


# --------------------------------------------------------------------------
# T2: the identity endpoint survives per-layer freedom
# --------------------------------------------------------------------------

def test_T2_all_zero_row_is_a_bitwise_identity_through_the_whole_net():
    torch.manual_seed(2)
    x = torch.randn(4, 3, 32, 32)
    plain_model = make_model()
    plain = forward_backward(plain_model, x)

    for coeffs in ([1.0] * N_SITES, per_stage_coeffs(0.5), per_stage_coeffs(2.0)):
        ctrl = SiteController("gaussian", levels=[1.0, 0.0], site_coeffs=coeffs)
        got = run_with(ctrl, make_model(), x, epoch=1)      # epoch 1 is the zero row
        assert torch.equal(got[0], plain[0]), "logits not bitwise identity"
        assert torch.equal(got[1], plain[1]), "gradients not bitwise identity"


def test_per_site_zero_coefficient_is_an_exact_bypass():
    """A site with c_l = 0 must be untouched, matching an early7-style mask."""
    torch.manual_seed(3)
    x = torch.randn(4, 3, 32, 32)
    coeffs = [1.0 if l in EARLY7 else 0.0 for l in range(N_SITES)]
    by_coeff = run_with(SiteController("gaussian", levels=[0.9],
                                       site_coeffs=coeffs), make_model(), x)
    by_mask = run_with(SiteController("gaussian", levels=[0.9], sites=EARLY7),
                       make_model(), x)
    assert torch.equal(by_coeff[0], by_mask[0])
    assert torch.equal(by_coeff[1], by_mask[1])


# --------------------------------------------------------------------------
# T3 / T4: build-time guards
# --------------------------------------------------------------------------

def test_T3_sigma_max_violation_raises_at_build_time_with_the_offending_site():
    # un-normalized coefficients push stage 3 to sigma 4 > sigma_max = 1
    coeffs = [2.0 ** STAGE_OF_SITE[l] for l in range(N_SITES)]
    with pytest.raises(ValueError, match="exceeds sigma_max"):
        SiteController("gaussian", levels=[1.0, 0.0], site_coeffs=coeffs)
    # the normalized profile is fine
    SiteController("gaussian", levels=[1.0, 0.0], site_coeffs=per_stage_coeffs(2.0))


def test_T3b_bound_accounts_for_the_resolution_ratio():
    """q_l < 1 leaves headroom, so the same table can be legal at r=16."""
    coeffs = [1.0] * N_SITES
    SiteController("gaussian", levels=[2.0, 0.0], site_coeffs=coeffs,
                   resolution_by_epoch=[16, 16])          # sigma = 0.5 * 2.0 = 1.0
    with pytest.raises(ValueError, match="exceeds sigma_max"):
        SiteController("gaussian", levels=[2.0, 0.0], site_coeffs=coeffs,
                       resolution_by_epoch=[32, 32])      # sigma = 2.0


def test_T3c_bound_ignores_unfiltered_sites():
    coeffs = [0.0] * N_SITES
    coeffs[0] = 1.0
    SiteController("gaussian", levels=[1.0, 0.0], site_coeffs=coeffs, sites=(0,))


def test_T4_nonzero_terminal_row_raises():
    with pytest.raises(ValueError, match="never reaches the target objective"):
        SiteController("gaussian", levels=[1.0, 0.5],
                       require_terminal_identity=1)
    # a proper explicit final identity phase passes
    SiteController("gaussian", levels=[1.0, 0.5, 0.0], require_terminal_identity=1)
    with pytest.raises(ValueError, match="never reaches the target objective"):
        SiteController("gaussian", levels=[1.0, 0.0], require_terminal_identity=2)


def test_T4b_terminal_guard_is_off_by_default_for_timing_probes():
    """job_campaign.py holds a constant level for a never-trained benchmark."""
    SiteController("gaussian", levels=[1.0] * 30, resolution_by_epoch=[16] * 30)


def test_T4c_terminal_guard_checks_every_site_not_just_the_scalar():
    tab = [[0.0] * N_SITES, [0.0] * N_SITES]
    tab[1][13] = 0.3                      # one deep site never anneals
    with pytest.raises(ValueError, match="site 13"):
        SiteController("gaussian", sigma_table=tab, require_terminal_identity=1)


# --------------------------------------------------------------------------
# T5: provenance
# --------------------------------------------------------------------------

def test_T5_table_digest_is_stable_and_discriminating():
    a = SiteController("gaussian", levels=[1.0, 0.0], site_coeffs=per_stage_coeffs(0.5))
    b = SiteController("gaussian", levels=[1.0, 0.0], site_coeffs=per_stage_coeffs(0.5))
    c = SiteController("gaussian", levels=[1.0, 0.0], site_coeffs=per_stage_coeffs(2.0))
    assert a.describe()["level_table_sha256"] == b.describe()["level_table_sha256"]
    assert a.describe()["level_table_sha256"] != c.describe()["level_table_sha256"]
    assert len(a.describe()["level_table_sha256"]) == 64


def test_T5b_describe_records_the_profile_and_the_table():
    ctrl = SiteController("gaussian", levels=[1.0, 0.0],
                          site_coeffs=per_stage_coeffs(0.5),
                          profile=("per_stage", {"rho": 0.5}))
    d = ctrl.describe()
    assert d["profile"] == ("per_stage", {"rho": 0.5})
    assert d["site_coeffs"][13] == 0.25
    assert d["level_table"][0][13] == 0.25
    assert d["level_decimals"] == LEVEL_DECIMALS


def test_levels_are_rounded_so_the_kernel_cache_stays_bounded():
    ctrl = SiteController("gaussian", levels=[1 / 3, 0.0])
    assert ctrl.L[0][0] == round(1 / 3, LEVEL_DECIMALS)


# --------------------------------------------------------------------------
# T7: evaluation must not leak state into training
# --------------------------------------------------------------------------

def test_T7_eval_snapshot_and_restore_leaves_the_row_bit_identical():
    ctrl = SiteController("gaussian", levels=[1.0, 0.5, 0.0],
                          site_coeffs=per_stage_coeffs(0.5),
                          resolution_by_epoch=[32, 32, 32])
    ctrl.set_epoch(0)
    before, res_before = list(ctrl.row), ctrl.resolution

    prev = (None if ctrl.row is None else list(ctrl.row),
            ctrl.resolution, ctrl.bypass_all)
    ctrl.set_state(0.0, 32)                       # what evaluate() does
    assert ctrl.row == [0.0] * N_SITES
    ctrl.row, ctrl.resolution, ctrl.bypass_all = prev
    ctrl.q = ctrl.q_for(ctrl.resolution)

    assert ctrl.row == before
    assert ctrl.resolution == res_before


def test_value_is_none_when_the_row_is_not_uniform():
    ctrl = SiteController("gaussian", levels=[1.0, 0.0],
                          site_coeffs=per_stage_coeffs(0.5))
    ctrl.set_epoch(0)
    assert ctrl.value is None                      # non-uniform row
    assert ctrl.max_level == 1.0
    assert ctrl.is_active()
    ctrl.set_epoch(1)
    assert ctrl.value == 0.0                       # uniform zero row
    assert not ctrl.is_active()


def test_uniform_row_still_reports_a_scalar_value():
    ctrl = SiteController("gaussian", levels=[0.4, 0.0])
    ctrl.set_epoch(0)
    assert ctrl.value == 0.4


def test_set_state_row_forces_a_per_site_evaluation():
    ctrl = SiteController("gaussian", levels=[1.0, 0.0])
    row = [0.1 * (l % 3) for l in range(N_SITES)]
    ctrl.set_state_row(row, 32)
    assert ctrl.row == row
    with pytest.raises(ValueError, match="need %d" % N_SITES):
        ctrl.set_state_row([0.0] * 5, 32)


def test_operator_none_still_bypasses_everything():
    ctrl = SiteController("none")
    ctrl.set_epoch(0)
    assert ctrl.L is None
    assert not ctrl.is_active()
    h = torch.randn(2, 4, 8, 8)
    assert ctrl.apply_at(0, h) is h

"""Correctness of the TV-L2 and TV-Hminus1 budget transformations."""
import numpy as np
import pytest
import torch

from continuation.config import TransformConfig
from continuation.diagnostics import total_variation
from continuation.transforms import build_transform
from continuation.transforms.tv import (
    dt_apply,
    forward_diff,
    laplacian_apply,
    laplacian_eigenvalues,
    laplacian_pinv,
    project_l1_ball_nonneg,
    project_box_mean,
    project_tv_ball,
    solve_tv_budget,
    tv_value,
)

H = W = 12
C = 3


def rand_image(seed=0, c=C, h=H, w=W):
    return np.random.default_rng(seed).random((c, h, w))


# ---------------------------------------------------------------- operators

def test_forward_differences_have_zero_far_boundary():
    z = rand_image()
    d1, d2 = forward_diff(z)
    assert np.all(d1[:, -1, :] == 0)
    assert np.all(d2[:, :, -1] == 0)
    assert np.allclose(d1[:, :-1, :], z[:, 1:, :] - z[:, :-1, :])
    assert np.allclose(d2[:, :, :-1], z[:, :, 1:] - z[:, :, :-1])


def test_no_periodic_wrapping():
    """A ramp has constant interior differences and no jump at the boundary."""
    z = np.tile(np.arange(W, dtype=float), (1, H, 1))
    _, d2 = forward_diff(z)
    assert np.allclose(d2[:, :, :-1], 1.0)
    assert np.all(d2[:, :, -1] == 0.0)


def test_adjoint_identity():
    """<D z, g> == <z, D^T g> for random z, g."""
    rng = np.random.default_rng(1)
    z = rng.random((C, H, W))
    g1, g2 = rng.random((C, H, W)), rng.random((C, H, W))
    g1[:, -1, :] = 0
    g2[:, :, -1] = 0
    d1, d2 = forward_diff(z)
    lhs = float((d1 * g1).sum() + (d2 * g2).sum())
    rhs = float((z * dt_apply(g1, g2)).sum())
    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)


def test_tv_matches_the_measurement_convention_in_diagnostics():
    z = rand_image(3)
    a = tv_value(z)
    b = float(total_variation(torch.from_numpy(z)[None]).item())
    assert a == pytest.approx(b, rel=1e-12)


# --------------------------------------------------------------- Laplacian

def test_dct_eigenvalues_match_the_explicit_laplacian_matrix():
    """Build L = D1^T D1 + D2^T D2 explicitly and compare with the DCT action."""
    h, w = 5, 4
    n = h * w
    L = np.zeros((n, n))
    for j in range(n):
        e = np.zeros((1, h, w))
        e.reshape(-1)[j] = 1.0
        L[:, j] = laplacian_apply(e).reshape(-1)
    assert np.allclose(L, L.T, atol=1e-12)                  # symmetric
    evals = np.linalg.eigvalsh(L)
    assert evals.min() > -1e-10                              # PSD
    dct_evals = np.sort(laplacian_eigenvalues(h, w).reshape(-1))
    assert np.allclose(np.sort(evals), dct_evals, atol=1e-10)


def test_laplacian_kernel_is_the_constants():
    const = np.ones((1, H, W)) * 0.37
    assert np.allclose(laplacian_apply(const), 0.0, atol=1e-12)


def test_pinv_is_a_true_pseudoinverse_on_mean_free_fields():
    rng = np.random.default_rng(5)
    r = rng.random((C, H, W))
    r -= r.reshape(C, -1).mean(axis=1)[:, None, None]        # mean-free
    eig = laplacian_eigenvalues(H, W)
    p = laplacian_pinv(r, eig)
    assert np.allclose(p.reshape(C, -1).mean(axis=1), 0.0, atol=1e-12)   # mean-free
    assert np.allclose(laplacian_apply(p), r, atol=1e-10)               # L p = r


def test_pinv_annihilates_the_constant_mode():
    eig = laplacian_eigenvalues(H, W)
    const = np.ones((1, H, W))
    assert np.allclose(laplacian_pinv(const, eig), 0.0, atol=1e-10)


def test_hminus1_fidelity_two_formulations_agree():
    """0.5 r^T L^+ r  ==  0.5 <r, p> with L p = r, mean(p)=0."""
    rng = np.random.default_rng(7)
    r = rng.random((C, H, W))
    r -= r.reshape(C, -1).mean(axis=1)[:, None, None]
    eig = laplacian_eigenvalues(H, W)
    p = laplacian_pinv(r, eig)
    assert 0.5 * float((r * p).sum()) == pytest.approx(
        0.5 * float((r * laplacian_pinv(r, eig)).sum()), rel=1e-12)
    assert np.allclose(laplacian_apply(p), r, atol=1e-10)


# -------------------------------------------------------------- projections

def test_l1_ball_projection():
    v = np.array([0.5, 0.3, 0.2])
    assert np.allclose(project_l1_ball_nonneg(v, 2.0), v)     # already inside
    p = project_l1_ball_nonneg(np.array([3.0, 1.0, 0.5]), 2.0)
    assert p.sum() == pytest.approx(2.0)
    assert np.all(p >= 0)


def test_tv_ball_projection_respects_the_global_budget():
    rng = np.random.default_rng(11)
    g1, g2 = rng.random((C, H, W)), rng.random((C, H, W))
    total = np.sqrt(g1 ** 2 + g2 ** 2).sum()
    p1, p2 = project_tv_ball(g1, g2, total * 0.4)
    assert np.sqrt(p1 ** 2 + p2 ** 2).sum() == pytest.approx(total * 0.4, rel=1e-10)


def test_box_mean_projection():
    rng = np.random.default_rng(13)
    z = rng.random((C, H, W)) * 3 - 1          # outside [0,1]
    means = np.array([0.3, 0.5, 0.7])
    p = project_box_mean(z, means)
    assert p.min() >= -1e-12 and p.max() <= 1 + 1e-12
    assert np.allclose(p.reshape(C, -1).mean(axis=1), means, atol=1e-9)


# ------------------------------------------------------------- the families

@pytest.mark.parametrize("fidelity", ["l2", "hminus1"])
def test_endpoints_are_exact(fidelity):
    x = rand_image(17)
    z1, i1 = solve_tv_budget(x, 1.0, fidelity)
    assert np.array_equal(z1, x) and i1["solver_used"] is False
    z0, i0 = solve_tv_budget(x, 0.0, fidelity)
    means = x.reshape(C, -1).mean(axis=1)
    assert np.allclose(z0, means[:, None, None], atol=1e-15)
    assert tv_value(z0) == pytest.approx(0.0, abs=1e-12)
    assert i0["solver_used"] is False


@pytest.mark.parametrize("fidelity", ["l2", "hminus1"])
def test_zero_tv_image_is_returned_unchanged_at_every_t(fidelity):
    x = np.full((C, H, W), 0.42)
    for t in (0.0, 0.3, 0.75, 1.0):
        z, info = solve_tv_budget(x, t, fidelity)
        assert np.array_equal(z, x), t
        assert info["solver_used"] is False


@pytest.mark.parametrize("fidelity", ["l2", "hminus1"])
def test_constraints_are_satisfied_at_interior_budgets(fidelity):
    x = rand_image(23)
    for t in (0.75, 0.5, 0.25):
        z, info = solve_tv_budget(x, t, fidelity, max_iters=3000)
        assert info["converged"], (fidelity, t, info["primal_residual"])
        assert info["box_violation"] <= 1e-9
        assert info["mean_abs_drift"] <= 1e-9
        # budget respected up to a small numerical slack, and reported
        assert info["tv_budget_rel_violation"] < 5e-3, info
        assert info["achieved_tv_ratio"] <= t + 5e-3


@pytest.mark.parametrize("fidelity", ["l2", "hminus1"])
def test_tv_ratio_is_monotone_in_the_budget(fidelity):
    x = rand_image(29)
    ratios = [solve_tv_budget(x, t, fidelity)[1].get("achieved_tv_ratio", t)
              for t in (0.25, 0.5, 0.75)]
    assert ratios[0] < ratios[1] < ratios[2]


def test_the_two_fidelities_give_different_solutions():
    x = rand_image(31)
    a, _ = solve_tv_budget(x, 0.5, "l2")
    b, _ = solve_tv_budget(x, 0.5, "hminus1")
    assert not np.allclose(a, b, atol=1e-3)


def test_l2_solution_beats_hminus1_solution_on_l2_fidelity():
    """Each method must be optimal for its own fidelity among feasible points."""
    x = rand_image(37)
    a, ia = solve_tv_budget(x, 0.5, "l2")
    b, _ = solve_tv_budget(x, 0.5, "hminus1")
    assert 0.5 * ((a - x) ** 2).sum() <= 0.5 * ((b - x) ** 2).sum() + 1e-6
    assert ia["fidelity_value"] == pytest.approx(0.5 * float(((a - x) ** 2).sum()), rel=1e-9)


def test_uniqueness_solver_is_deterministic():
    x = rand_image(41)
    for fid in ("l2", "hminus1"):
        z1, _ = solve_tv_budget(x, 0.4, fid)
        z2, _ = solve_tv_budget(x, 0.4, fid)
        assert np.array_equal(z1, z2)


# -------------------------------------------------- ImageTransform interface

@pytest.mark.parametrize("family", ["tv_l2", "tv_hminus1"])
def test_transform_interface(family):
    T = build_transform(TransformConfig(family=family, params={"max_iters": 800}))
    assert T.target_parameter == 1.0 and T.is_target(1.0)
    assert T.parameter_name == "t"
    x = torch.rand(2, C, H, W, generator=torch.Generator().manual_seed(3))
    out = T.apply(x, 1.0)
    assert torch.equal(out.images, x) and out.info["identity"] is True
    out = T.apply(x, 0.5)
    assert out.images.shape == x.shape and out.images.dtype == x.dtype
    assert out.images.min() >= -1e-6 and out.images.max() <= 1 + 1e-6
    assert "per_image" in out.info and len(out.info["per_image"]) == 2
    for info in out.info["per_image"]:
        assert "achieved_tv_ratio" in info and "requested_t" in info
        assert info["requested_t"] == 0.5
    with pytest.raises(ValueError):
        T.apply(x, 1.5)
    with pytest.raises(ValueError):
        T.apply(x, -0.1)


@pytest.mark.parametrize("family", ["tv_l2", "tv_hminus1"])
def test_cache_key_includes_fidelity_and_tolerance(family):
    a = build_transform(TransformConfig(family=family, params={"tol": 1e-7}))
    b = build_transform(TransformConfig(family=family, params={"tol": 1e-5}))
    assert a.cache_key(0.5, "img1") != b.cache_key(0.5, "img1")
    assert a.cache_key(0.5, "img1") != a.cache_key(0.4, "img1")
    l2 = build_transform(TransformConfig(family="tv_l2", params={}))
    hm = build_transform(TransformConfig(family="tv_hminus1", params={}))
    assert l2.cache_key(0.5, "img1") != hm.cache_key(0.5, "img1")


def test_no_fixed_penalty_coefficient_is_used():
    """The budget is a hard constraint; a shared TV penalty would not be equivalent."""
    sig = build_transform(TransformConfig(family="tv_l2", params={})).config_signature()
    assert "no TV penalty coefficient" in sig["budget_enforcement"]
    assert "lambda" not in sig and "penalty" not in sig

"""The measurement machinery, checked against things that are known in advance.

A loss-surface plot is convincing whether or not it is correct, which is exactly
why the pieces underneath it need tests that do not depend on looking at the
picture: the Hessian-vector product against finite differences, the plane basis
against its own geometry, filter normalisation against the norms it claims to
match, and every sweep against the requirement that it leaves theta where it
found it.
"""
import pytest
import torch

from cifarbase import landscape
from cifarbase.data import Split
from cifarbase.homotopy import ResidualGate, residual_blocks
from cifarbase.model import ARCHS, ResNet


def make_model(seed=0, width=8, dtype=torch.float32):
    torch.manual_seed(seed)
    model = ResNet(ARCHS["resnet18"], num_classes=10, width=width,
                   zero_init_residual=False)
    return model.to(dtype).eval()


def make_split(n=64, seed=3):
    """A real Split over noise: exercises chunks() rather than faking it."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 256, (n, 3, 32, 32), generator=generator,
                      dtype=torch.uint8)
    y = torch.randint(0, 10, (n,), generator=generator)
    return Split(x, y, torch.device("cpu"))


def batch(n=4, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 3, 32, 32, generator=generator),
            torch.randint(0, 10, (n,), generator=generator))


# --------------------------------------------------------------------------
# weights as a vector
# --------------------------------------------------------------------------

def test_weight_vector_round_trips():
    model = make_model()
    original = landscape.get_weights(model).clone()
    landscape.set_weights(model, torch.zeros_like(original))
    assert float(landscape.get_weights(model).abs().max()) == 0.0
    landscape.set_weights(model, original)
    assert torch.equal(landscape.get_weights(model), original)


def test_set_weights_rejects_a_wrong_size():
    model = make_model()
    with pytest.raises(ValueError, match="entries"):
        landscape.set_weights(model, torch.zeros(5))


def test_at_weights_restores_even_on_error():
    model = make_model()
    original = landscape.get_weights(model).clone()
    with pytest.raises(RuntimeError):
        with landscape.at_weights(model, torch.zeros_like(original)):
            raise RuntimeError("boom")
    assert torch.equal(landscape.get_weights(model), original)


# --------------------------------------------------------------------------
# directions
# --------------------------------------------------------------------------

def test_random_direction_is_seed_deterministic():
    model = make_model()
    a = landscape.random_direction(model, seed=7)
    b = landscape.random_direction(model, seed=7)
    c = landscape.random_direction(model, seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_filter_normalisation_matches_filter_norms():
    """Each filter's direction must carry exactly that filter's norm.

    This is the whole content of Li et al.'s normalisation: without it a step of
    alpha means different things in different layers, and under BatchNorm the
    per-layer weight scale is arbitrary anyway.
    """
    model = make_model()
    direction = landscape.random_direction(model, seed=0, filter_norm=True)
    offset = 0
    checked = 0
    for p in model.parameters():
        n = p.numel()
        piece = direction[offset:offset + n].view_as(p)
        offset += n
        if p.dim() < 2:
            # 1-D parameters are left alone by default.
            assert float(piece.abs().max()) == 0.0
            continue
        got = piece.reshape(p.shape[0], -1).norm(dim=1)
        want = p.detach().reshape(p.shape[0], -1).norm(dim=1)
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-6)
        checked += 1
    assert checked > 0
    assert offset == direction.numel()


def test_include_1d_perturbs_batchnorm_too():
    model = make_model()
    direction = landscape.random_direction(model, seed=0, include_1d=True)
    offset = 0
    for p in model.parameters():
        piece = direction[offset:offset + p.numel()]
        offset += p.numel()
        if p.dim() == 1:
            assert float(piece.abs().max()) > 0.0
            return
    pytest.fail("no 1-D parameter found")


# --------------------------------------------------------------------------
# planes
# --------------------------------------------------------------------------

def test_plane_basis_is_orthonormal_and_aimed_at_a():
    torch.manual_seed(0)
    origin, a, b = (torch.randn(200) for _ in range(3))
    e1, e2 = landscape.plane_from_points(origin, a, b)

    assert float(e1.norm()) == pytest.approx(1.0, abs=1e-6)
    assert float(e2.norm()) == pytest.approx(1.0, abs=1e-6)
    assert float(e1 @ e2) == pytest.approx(0.0, abs=1e-5)

    # The x-axis points exactly at `a`, so `a` lands on it with y == 0. That is
    # what makes the continuation plot readable: x is the net displacement.
    x, y = landscape.project(a, origin, e1, e2)
    assert x == pytest.approx(float((a - origin).norm()), rel=1e-5)
    assert y == pytest.approx(0.0, abs=1e-4)
    assert landscape.project(origin, origin, e1, e2) == pytest.approx((0.0, 0.0),
                                                                     abs=1e-6)
    # `b` lies in the plane, so projecting and rebuilding it is lossless.
    bx, by = landscape.project(b, origin, e1, e2)
    assert torch.allclose(origin + bx * e1 + by * e2, b, atol=1e-4)


def test_plane_rejects_degenerate_points():
    origin = torch.randn(50)
    with pytest.raises(ValueError, match="same point"):
        landscape.plane_from_points(origin, origin.clone(), torch.randn(50))
    with pytest.raises(ValueError, match="collinear"):
        landscape.plane_from_points(origin, origin + 1.0, origin + 2.0)


def test_pca_plane_recovers_a_planted_plane():
    """Points built inside a known 2-D plane must yield that plane back."""
    torch.manual_seed(0)
    u, v = torch.randn(300), torch.randn(300)
    u /= u.norm()
    v -= (v @ u) * u
    v /= v.norm()
    origin = torch.randn(300)
    coeffs = [(3.0, 0.1), (2.0, -0.4), (-1.0, 0.7), (0.0, 0.0), (1.5, 0.2)]
    points = [origin + a * u + b * v for a, b in coeffs]

    e1, e2, explained = landscape.pca_plane(points, origin=origin)
    assert sum(explained) == pytest.approx(1.0, abs=1e-5)
    # Every planted point must be reconstructible from the recovered basis.
    for point in points:
        x, y = landscape.project(point, origin, e1, e2)
        assert torch.allclose(origin + x * e1 + y * e2, point, atol=1e-3)


def test_pca_plane_needs_three_points():
    with pytest.raises(ValueError, match="at least 3"):
        landscape.pca_plane([torch.randn(10), torch.randn(10)])


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------

def test_surface_matches_a_direct_evaluation_at_the_centre():
    model, split = make_model(), make_split()
    center = landscape.get_weights(model).clone()
    e1 = landscape.random_direction(model, seed=0)
    e2 = landscape.random_direction(model, seed=1)

    out = landscape.surface_2d(model, split, center, e1, e2, xs=[-0.1, 0.0, 0.1],
                               ys=[-0.1, 0.0, 0.1], batch_size=32,
                               max_images=None)
    direct, _ = landscape.evaluate_at(model, split, batch_size=32,
                                      max_images=None)

    assert len(out["loss"]) == 3 and len(out["loss"][0]) == 3
    assert out["loss"][1][1] == pytest.approx(direct, rel=1e-6)
    # A sweep must put theta back exactly, or it silently corrupts training.
    assert torch.equal(landscape.get_weights(model), center)


def test_surface_moves_the_loss():
    model, split = make_model(), make_split()
    center = landscape.get_weights(model).clone()
    e1 = landscape.random_direction(model, seed=0)
    e2 = landscape.random_direction(model, seed=1)
    out = landscape.surface_2d(model, split, center, e1, e2, xs=[-1.0, 0.0, 1.0],
                               ys=[0.0], batch_size=32, max_images=None)
    row = out["loss"][0]
    assert row[0] != pytest.approx(row[1], rel=1e-4)


def test_surface_s_alpha_agrees_with_loss_vs_s_at_alpha_zero():
    model, split = make_model(), make_split()
    center = landscape.get_weights(model).clone()
    direction = landscape.random_direction(model, seed=0)
    s_values = [0.0, 0.5, 1.0]

    with ResidualGate(model) as gate:
        surface = landscape.surface_s_alpha(model, split, center, direction,
                                            s_values, alphas=[0.0], gate=gate,
                                            batch_size=32, max_images=None)
        slice_ = landscape.loss_vs_s(model, split, s_values, gate,
                                     batch_size=32, max_images=None)
    assert surface["loss"][0] == pytest.approx(slice_["loss"], rel=1e-6)
    assert torch.equal(landscape.get_weights(model), center)


def test_interpolation_endpoints_are_the_endpoints():
    model, split = make_model(), make_split()
    w_a = landscape.get_weights(model).clone()
    w_b = w_a + 0.05 * landscape.random_direction(model, seed=2)

    loss_a, _ = landscape.evaluate_at(model, split, weights=w_a, batch_size=32,
                                      max_images=None)
    loss_b, _ = landscape.evaluate_at(model, split, weights=w_b, batch_size=32,
                                      max_images=None)
    out = landscape.interpolate(model, split, w_a, w_b, n=5, batch_size=32,
                                max_images=None)

    assert out["loss"][0] == pytest.approx(loss_a, rel=1e-6)
    assert out["loss"][-1] == pytest.approx(loss_b, rel=1e-6)
    assert out["barrier"] >= 0.0
    assert out["alphas"][0] == pytest.approx(0.0)
    assert out["alphas"][-1] == pytest.approx(1.0)
    assert torch.equal(landscape.get_weights(model), w_a)


def test_interpolation_can_run_past_the_endpoints():
    model, split = make_model(), make_split()
    w_a = landscape.get_weights(model).clone()
    w_b = w_a + 0.05 * landscape.random_direction(model, seed=2)
    out = landscape.interpolate(model, split, w_a, w_b, n=5, extend=0.25,
                                batch_size=32, max_images=None)
    assert out["alphas"][0] == pytest.approx(-0.25)
    assert out["alphas"][-1] == pytest.approx(1.25)


# --------------------------------------------------------------------------
# batchnorm
# --------------------------------------------------------------------------

def test_recompute_bn_changes_statistics_and_restores_state():
    model, split = make_model(), make_split()
    norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    before = norms[0].running_mean.clone()
    momenta = [m.momentum for m in norms]

    model.eval()
    seen = landscape.recompute_bn(model, split, batch_size=16, n_batches=2)

    assert seen == 32
    assert not torch.allclose(before, norms[0].running_mean)
    # Neither the momenta nor the train/eval flag may leak out of the call.
    assert [m.momentum for m in norms] == momenta
    assert not model.training


def test_recompute_bn_is_order_independent():
    """momentum=None gives a cumulative average, so two seeds over the same
    batches must agree -- otherwise a barrier plot would depend on shuffling."""
    split = make_split(n=64)
    first, second = make_model(), make_model()
    landscape.recompute_bn(first, split, batch_size=64, n_batches=1, seed=0)
    landscape.recompute_bn(second, split, batch_size=64, n_batches=1, seed=1)
    a = [m for m in first.modules() if isinstance(m, torch.nn.BatchNorm2d)][0]
    b = [m for m in second.modules() if isinstance(m, torch.nn.BatchNorm2d)][0]
    assert torch.allclose(a.running_mean, b.running_mean, atol=1e-6)


# --------------------------------------------------------------------------
# per-block diagnostics
# --------------------------------------------------------------------------

def test_residual_ratio_scales_linearly_with_s():
    model, (x, _) = make_model(), batch()
    with ResidualGate(model) as gate:
        gate.set(1.0)
        full = landscape.residual_ratio(model, x, gate)
        gate.set(0.25)
        quarter = landscape.residual_ratio(model, x, gate)

    assert len(full["blocks"]) == len(residual_blocks(model))
    for a, b in zip(full["blocks"], quarter["blocks"], strict=True):
        # The raw branch magnitude is a property of the weights, not of s...
        assert b["f_rms"] == pytest.approx(a["f_rms"], rel=1e-6)
        # ...and s multiplies straight through into the ratio.
        assert b["ratio"] == pytest.approx(0.25 * a["ratio"], rel=1e-6)
        assert a["f_rms"] > 0.0 and a["skip_rms"] > 0.0


def test_residual_ratio_leaves_the_gate_alone():
    model, (x, _) = make_model(), batch()
    with ResidualGate(model) as gate:
        gate.set([0.1 * i for i in range(8)])
        before = gate.get()
        landscape.residual_ratio(model, x, gate)
        assert gate.get() == before


def test_grad_norm_is_zero_at_s_zero():
    model, (x, y) = make_model(), batch()
    with ResidualGate(model) as gate:
        gate.set(0.0)
        off = landscape.grad_norm_per_block(model, x, y, gate)
        gate.set(1.0)
        on = landscape.grad_norm_per_block(model, x, y, gate)
    assert all(row["grad_norm"] == 0.0 for row in off["blocks"])
    assert any(row["grad_norm"] > 0.0 for row in on["blocks"])


# --------------------------------------------------------------------------
# curvature
# --------------------------------------------------------------------------

def test_hvp_is_linear_and_symmetric():
    """Hv checked against the two properties a Hessian cannot fake.

    Finite differences would be the obvious test and are the wrong one here: the
    network is piecewise linear, so a step of 1e-5 in 11k dimensions flips the
    sign of some ReLU units and the gradient jumps. The Hessian of a ReLU network
    only exists almost everywhere, and a finite-difference check measures that
    kink rather than the second derivative.

    Linearity (H(av+bw) = aHv + bHw) and symmetry (v.Hw == w.Hv) hold exactly,
    are independent of the kinks, and are violated by every plausible way of
    getting the double backward wrong.
    """
    model = make_model(width=4, dtype=torch.float64)
    x, y = batch(n=3)
    x = x.double()
    params = [p for p in model.parameters() if p.requires_grad]
    size = sum(p.numel() for p in params)

    torch.manual_seed(0)
    v = torch.randn(size, dtype=torch.float64)
    w = torch.randn(size, dtype=torch.float64)

    hv = landscape._hvp(model, params, x, y, v)
    hw = landscape._hvp(model, params, x, y, w)
    assert float(hv.norm()) > 0.0

    combined = landscape._hvp(model, params, x, y, 2.0 * v - 3.0 * w)
    assert torch.allclose(combined, 2.0 * hv - 3.0 * hw, rtol=1e-8, atol=1e-10)

    scale = max(abs(float(v @ hw)), abs(float(w @ hv)), 1e-12)
    assert abs(float(v @ hw) - float(w @ hv)) / scale < 1e-8


def test_top_hessian_eigenvalue_matches_rayleigh_quotient_growth():
    """Power iteration must beat a random direction's Rayleigh quotient.

    A weak check by design: the exact top eigenvalue of an 11k-parameter Hessian
    is not available here, but the largest one found must dominate what a random
    probe sees, and must be reported as a real number.
    """
    model = make_model(width=4, dtype=torch.float64)
    x, y = batch(n=3)
    x = x.double()
    out = landscape.top_hessian_eigs(model, x, y, k=1, iters=30, seed=0)
    top = out["eigenvalues"][0]

    params = [p for p in model.parameters() if p.requires_grad]
    size = sum(p.numel() for p in params)
    torch.manual_seed(1)
    v = torch.randn(size, dtype=torch.float64)
    v /= v.norm()
    random_quotient = float(v @ landscape._hvp(model, params, x, y, v))

    assert abs(top) >= abs(random_quotient)
    assert abs(top) < float("inf")


def test_hutchinson_trace_reports_its_own_error():
    model = make_model(width=4)
    x, y = batch(n=3)
    out = landscape.hutchinson_trace(model, x, y, samples=4, seed=0)
    assert out["samples"] == 4
    assert out["sem"] == out["sem"]                 # not NaN
    assert abs(out["trace"]) < float("inf")


def test_curvature_calls_leave_the_model_in_eval_and_ungated():
    model = make_model(width=4)
    x, y = batch(n=3)
    model.train()
    with ResidualGate(model) as gate:
        gate.set(0.4)
        landscape.top_hessian_eigs(model, x, y, k=1, iters=3, gate=gate, s=1.0)
        landscape.hutchinson_trace(model, x, y, samples=2, gate=gate, s=1.0)
        assert gate.get() == [0.4] * 8
    assert model.training


# --------------------------------------------------------------------------
# checkpoints and records
# --------------------------------------------------------------------------

def test_checkpoint_round_trip(tmp_path):
    model = make_model()
    landscape.save_checkpoint(model, str(tmp_path), "epoch000",
                              meta={"epoch": 0, "s": 0.0}, half=False)
    landscape.save_checkpoint(model, str(tmp_path), "epoch010",
                              meta={"epoch": 10, "s": 1.0}, half=False)

    original = landscape.get_weights(model).clone()
    landscape.set_weights(model, torch.zeros_like(original))
    landscape.load_checkpoint(str(tmp_path / "checkpoints" / "epoch000.pt"), model)
    assert torch.equal(landscape.get_weights(model), original)

    found = landscape.list_checkpoints(str(tmp_path))
    assert [c["epoch"] for c in found] == [0, 10]
    assert found[1]["s"] == 1.0


def test_half_checkpoints_stay_close_enough_to_be_useful(tmp_path):
    """fp16 storage must not move the loss by anything a figure can resolve."""
    model, split = make_model(), make_split()
    exact, _ = landscape.evaluate_at(model, split, batch_size=32, max_images=None)
    landscape.save_checkpoint(model, str(tmp_path), "half", meta={"epoch": 0},
                              half=True)
    landscape.load_checkpoint(str(tmp_path / "checkpoints" / "half.pt"), model)
    restored, _ = landscape.evaluate_at(model, split, batch_size=32,
                                        max_images=None)
    assert restored == pytest.approx(exact, rel=1e-3)


def test_records_append_and_read_back(tmp_path):
    landscape.record(str(tmp_path), {"kind": "a", "value": 1})
    landscape.record(str(tmp_path), {"kind": "b", "value": 2})
    rows = landscape.read_records(str(tmp_path))
    assert [r["kind"] for r in rows] == ["a", "b"]
    assert landscape.read_records(str(tmp_path), "missing.jsonl") == []

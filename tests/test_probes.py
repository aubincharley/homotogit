"""The added probes, and the guards that decide whether their numbers are readable."""
from __future__ import annotations

import math

import pytest
import torch

from continuation_core.analysis import curvature, gauge, power, sensitivity


# -- guards ----------------------------------------------------------------

def test_validity_gate_rejects_a_mismatched_cell():
    r = power.validity_gate({"good": 0.2000, "bad": 0.3872},
                            {"good": 0.2000, "bad": 0.2149})
    assert r["admitted"] == ["good"]
    assert pytest.approx(r["rejected"]["bad"], abs=1e-6) == 0.1723


def test_detection_ceiling_reproduces_the_two_measured_cases():
    # 500-image training probe: the gap was almost entirely sampling noise
    assert power.detection_ceiling(0.0143, 0.0139)["ceiling"] == pytest.approx(0.235, abs=0.01)
    # whole training subset: the same test becomes able to answer
    assert power.detection_ceiling(0.0088, 0.0040)["ceiling"] == pytest.approx(0.891, abs=0.01)


def test_noise_floor_is_the_paired_difference_over_sqrt_two():
    pairs = [(1.0, 1.2), (2.0, 2.1), (3.0, 3.4), (4.0, 4.1)]
    d = [b - a for a, b in pairs]
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    assert power.noise_floor(pairs)["sd"] == pytest.approx(sd / math.sqrt(2))


def test_partial_correlation_removes_a_common_driver():
    """x and y both driven by z, plus independent jitter.

    Exact collinearity would make the partial correlation *undefined*, not zero,
    so the driver carries noise -- which is also the realistic case: ``||J||_F``
    scored +0.81 against the gap and +0.22 once the training error was held
    fixed, and that is the pattern this reproduces.
    """
    torch.manual_seed(0)
    z = torch.linspace(0, 5, 40)
    x = (2 * z + 0.3 * torch.randn(40)).tolist()
    y = (-3 * z + 0.3 * torch.randn(40)).tolist()
    z = z.tolist()
    assert power.pearson(x, y) < -0.9          # strong, and entirely spurious
    assert abs(power.partial(x, y, z)) < 0.4   # collapses once z is held fixed


def test_spearman_ignores_a_single_distant_point_that_pearson_follows():
    """One distant point manufactures a Pearson coefficient the ranks deny.

    This is not hypothetical: a curvature-versus-gap correlation read +0.94 in
    Pearson and +0.44 on ranks, the whole difference being one arm far from the
    cloud.  Both are now always reported.
    """
    x = list(range(1, 13)) + [1000]
    y = list(range(12, 0, -1)) + [1000]
    assert power.pearson(x, y) > 0.9          # manufactured by the outlier
    assert power.spearman(x, y) < -0.5        # the monotone truth on the rest


# -- gauge -----------------------------------------------------------------

def _tiny_state():
    torch.manual_seed(0)
    return {"conv1.weight": torch.randn(4, 3, 3, 3),
            "bn1.weight": torch.ones(4), "bn1.bias": torch.zeros(4),
            "bn1.running_mean": torch.randn(4), "bn1.running_var": torch.rand(4) + 0.5,
            "fc.weight": torch.randn(2, 4), "fc.bias": torch.zeros(2)}


def test_conv_bn_pairs_finds_the_pair_and_rejects_a_convolution_without_bn():
    st = _tiny_state()
    assert gauge.conv_bn_pairs(st) == [("conv1.weight", "bn1")]
    st["conv2.weight"] = torch.randn(4, 4, 3, 3)      # no bn2 behind it
    assert gauge.conv_bn_pairs(st) == [("conv1.weight", "bn1")]


def test_rescaling_leaves_the_eval_mode_output_unchanged():
    st = _tiny_state()
    x = torch.randn(8, 4, 5, 5)

    def bn(s):
        return (torch.nn.functional.batch_norm(
            x * 1.0, s["bn1.running_mean"], s["bn1.running_var"],
            s["bn1.weight"], s["bn1.bias"], training=False))

    before = bn(st)
    out = {k: v.clone() for k, v in st.items()}
    alpha = torch.tensor([2.0, 0.5, 3.0, 1.5])
    gauge.rescale_(out, "conv1.weight", "bn1", alpha)
    after = torch.nn.functional.batch_norm(
        x * alpha.view(1, -1, 1, 1), out["bn1.running_mean"], out["bn1.running_var"],
        out["bn1.weight"], out["bn1.bias"], training=False)
    # float32 through a per-filter scale of up to 3; the equivalence is exact in
    # exact arithmetic, and this is the accumulation error, not a gauge error
    assert torch.allclose(before, after, atol=1e-4)
    assert float((before - after).abs().max()) < 1e-4


def test_a_negative_scale_is_refused_because_it_is_not_a_symmetry():
    st = _tiny_state()
    with pytest.raises(ValueError):
        gauge.rescale_(st, "conv1.weight", "bn1", torch.tensor([1.0, -1.0, 1.0, 1.0]))


def test_gauge_minimal_distance_is_never_larger_than_the_raw_one():
    a, b = _tiny_state(), _tiny_state()
    a["conv1.weight"] = a["conv1.weight"] * 1.7       # pure gauge move
    learn = ["conv1.weight", "bn1.weight", "bn1.bias", "fc.weight", "fc.bias"]
    d = gauge.distance_breakdown(a, b, learn)
    assert d["distance_gauge_minimal"] <= d["distance_raw"] + 1e-9
    assert d["distance_gauge_minimal"] < 1e-5         # the move was entirely gauge


# -- function-side probes ---------------------------------------------------

def test_singular_spectrum_matches_a_reference_svd():
    torch.manual_seed(0)
    J = torch.randn(3, 4, 2, 5, 5)
    got = sensitivity.singular_spectrum(J)
    want = torch.linalg.svdvals(J.reshape(3, 4, -1).to(torch.float64))
    assert torch.allclose(got, want, atol=1e-8)


def test_top_singular_direction_is_unit_and_maximises_the_response():
    torch.manual_seed(0)
    J = torch.randn(2, 4, 3, 4, 4)
    v = sensitivity.top_singular_direction(J)
    assert torch.allclose(v.reshape(2, -1).norm(dim=1), torch.ones(2), atol=1e-6)
    flat = J.reshape(2, 4, -1).to(torch.float64)
    best = torch.einsum("bki,bi->bk", flat, v.reshape(2, -1).to(torch.float64)).norm(dim=1)
    smax = sensitivity.singular_spectrum(J)[:, 0]
    assert torch.allclose(best, smax, atol=1e-6)


def test_radial_bins_cover_every_mode_exactly_once():
    idx, n = sensitivity.radial_bins(8, torch.device("cpu"))
    assert idx.shape == (8, 8)
    assert int(torch.bincount(idx.reshape(-1), minlength=n).sum()) == 64


def test_frequency_profile_is_a_normalised_distribution():
    torch.manual_seed(0)
    J = torch.randn(2, 3, 3, 8, 8)
    p = sensitivity.frequency_profile(J)
    assert sum(p["ring_energy_fraction"]) == pytest.approx(1.0, abs=1e-9)
    assert sum(p["ring_mode_count"]) == 64


def test_block_names_split_stem_blocks_and_classifier():
    assert curvature.block_of("conv1.weight") == "stem"
    assert curvature.block_of("blocks.3.conv2.weight") == "block3"
    assert curvature.block_of("fc.bias") == "fc"


def test_hessian_vector_product_is_exact_on_a_quadratic():
    """A linear model with squared error has a constant, known Hessian."""
    torch.manual_seed(0)
    w = torch.zeros(3, requires_grad=True)
    A = torch.randn(5, 3)

    def hvp(v):
        loss = ((A @ w) ** 2).sum()
        g, = torch.autograd.grad(loss, [w], create_graph=True)
        out, = torch.autograd.grad((g * v).sum(), [w])
        return out

    H = 2 * A.T @ A
    for _ in range(3):
        v = torch.randn(3)
        assert torch.allclose(hvp(v), H @ v, atol=1e-5)

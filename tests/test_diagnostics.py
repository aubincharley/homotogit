import math

import pytest
import torch

from continuation.diagnostics import (
    compare_gradients,
    per_image_mse,
    retained_tv_ratio,
    summarize_across_seeds,
    total_variation,
    transform_statistics,
)
from continuation.transforms import GaussianSmoothing


def test_tv_of_constant_image_is_zero():
    x = torch.full((3, 3, 8, 8), 0.37)
    assert torch.allclose(total_variation(x), torch.zeros(3, dtype=torch.float64), atol=1e-12)


def test_tv_of_a_single_vertical_step_matches_hand_calculation():
    """One channel, 4x4, left half 1.0 and right half 0.0.

    With forward differences and a zero normal difference at the last column,
    the only non-zero gradients are at column 1 (magnitude 1) in every row.
    """
    x = torch.zeros(1, 1, 4, 4)
    x[..., :2] = 1.0
    assert total_variation(x).item() == pytest.approx(4.0)


def test_tv_isotropy_couples_the_two_directions_per_pixel():
    """A diagonal step gives sqrt(2) per corner pixel, not 2."""
    x = torch.zeros(1, 1, 2, 2)
    x[0, 0, 0, 0] = 1.0
    # d1[0,0] = -1, d2[0,0] = -1 -> sqrt(2); d1[0,1]=0,d2 last col=0;
    # d1 last row = 0; d2[1,0] = 0 -> total sqrt(2)
    assert total_variation(x).item() == pytest.approx(math.sqrt(2.0))


def test_tv_sums_over_channels():
    x = torch.zeros(1, 3, 4, 4)
    x[:, :, :, :2] = 1.0
    assert total_variation(x).item() == pytest.approx(12.0)


def test_smoothing_reduces_tv_and_ratio_is_in_range():
    T = GaussianSmoothing(sigma_max=3.0)
    g = torch.Generator().manual_seed(1)
    x = torch.rand(4, 3, 32, 32, generator=g)
    prev = total_variation(x)
    for sigma in (0.5, 1.0, 2.0, 3.0):
        tv = total_variation(T(x, sigma))
        assert torch.all(tv < prev)
        prev = tv
    ratio, valid = retained_tv_ratio(T(x, 1.0), x)
    assert bool(valid.all())
    assert torch.all((ratio > 0) & (ratio < 1))


def test_zero_tv_images_are_marked_not_silently_set_to_one():
    x = torch.full((2, 3, 8, 8), 0.5)
    ratio, valid = retained_tv_ratio(x, x)
    assert not bool(valid.any())
    assert torch.isnan(ratio).all()


def test_per_image_mse_and_identity():
    g = torch.Generator().manual_seed(2)
    x = torch.rand(3, 3, 8, 8, generator=g)
    assert torch.allclose(per_image_mse(x, x), torch.zeros(3, dtype=torch.float64), atol=1e-12)
    y = x + 0.1
    assert torch.allclose(per_image_mse(y, x), torch.full((3,), 0.01, dtype=torch.float64), atol=1e-9)


def test_transform_statistics_reports_identity_at_the_target():
    T = GaussianSmoothing(sigma_max=3.0)
    g = torch.Generator().manual_seed(4)
    x = torch.rand(8, 3, 32, 32, generator=g)
    s0 = transform_statistics(T, 0.0, x)
    assert s0["is_target_endpoint"] is True
    assert s0["reconstruction_mse"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert s0["retained_tv_ratio"]["mean"] == pytest.approx(1.0, abs=1e-12)
    s2 = transform_statistics(T, 2.0, x)
    assert s2["reconstruction_mse"]["mean"] > s0["reconstruction_mse"]["mean"]
    assert s2["retained_tv_ratio"]["mean"] < 1.0
    assert s2["n_zero_tv_images"] == 0
    assert s2["tv_convention"]["differences"] == "forward"


def test_compare_gradients_reports_norms_and_cosine():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 2.0, 0.0])
    out = compare_gradients(a, b)
    assert out["cosine_available"] is True
    assert out["cosine"] == pytest.approx(0.0, abs=1e-7)
    assert out["grad_norm_old"] == pytest.approx(1.0)
    assert out["grad_norm_new"] == pytest.approx(2.0)
    assert out["grad_diff_norm"] == pytest.approx(math.sqrt(5.0))


def test_cosine_is_unavailable_near_a_stationary_point():
    tiny = torch.tensor([1e-12, 0.0, 0.0])
    out = compare_gradients(tiny, torch.tensor([1.0, 0.0, 0.0]))
    assert out["cosine"] is None
    assert out["cosine_available"] is False
    assert "stationary point" in out["cosine_unavailable_reason"]
    assert out["grad_norm_old"] < 1e-10        # norms still reported


def test_summarize_across_seeds():
    s = summarize_across_seeds([1.0, 2.0, 3.0])
    assert s["n"] == 3 and s["mean"] == pytest.approx(2.0)
    assert s["std"] == pytest.approx(1.0)
    assert s["sem"] == pytest.approx(1.0 / math.sqrt(3))
    assert summarize_across_seeds([])["mean"] is None
    assert summarize_across_seeds([5.0])["sem"] is None

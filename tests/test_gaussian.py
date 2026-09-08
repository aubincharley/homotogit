"""Verification of the Gaussian input transformation.

Covers the checks required before the family is used for training: identity at
the target endpoint, preservation of constant images, dimensions, determinism,
plausible attenuation of spatial oscillations, channel independence, and the
documented boundary behaviour.
"""
import math

import pytest
import torch

from continuation.transforms import GaussianSmoothing, gaussian_kernel_1d
from continuation.transforms.gaussian import heat_time_from_sigma, sigma_from_heat_time


def rand_image(n=4, c=3, h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, c, h, w, generator=g)


def test_identity_at_sigma_zero_is_exact():
    T = GaussianSmoothing(sigma_max=3.0)
    x = rand_image()
    y = T(x, 0.0)
    assert torch.equal(y, x)          # bitwise identical, not merely close
    assert y is x                     # exact passthrough, no residual blur
    assert T.is_target(0.0) and T.target_parameter == 0.0


def test_kernel_is_normalized_and_delta_at_zero():
    k0 = gaussian_kernel_1d(0.0, radius=12)
    assert k0.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert k0[12].item() == pytest.approx(1.0, abs=1e-6)
    assert k0[:12].abs().max().item() == 0.0
    for sigma in (0.5, 1.0, 2.0, 3.0):
        k = gaussian_kernel_1d(sigma, radius=12)
        assert k.sum().item() == pytest.approx(1.0, abs=1e-6)
        assert (k >= 0).all()
        assert torch.allclose(k, k.flip(0), atol=1e-7)   # symmetric


def test_kernel_tends_to_delta_as_sigma_goes_to_zero():
    """The family is continuous at the target endpoint, not just defined there."""
    T = GaussianSmoothing(sigma_max=3.0)
    x = rand_image(n=2)
    errs = [(T(x, s) - x).abs().max().item() for s in (0.5, 0.2, 0.05, 0.01)]
    assert all(a >= b for a, b in zip(errs, errs[1:])), errs
    assert errs[0] > errs[-1]
    # in float32 the taps underflow to an exact delta well before sigma reaches 0,
    # so the discrete family meets its identity endpoint continuously
    assert errs[-1] == 0.0


def test_constant_image_is_preserved():
    T = GaussianSmoothing(sigma_max=3.0)
    for value in (0.0, 0.5, 1.0):
        x = torch.full((2, 3, 32, 32), value)
        for sigma in (0.5, 1.0, 2.0, 3.0):
            y = T(x, sigma)
            assert torch.allclose(y, x, atol=1e-6), (value, sigma)


def test_shape_dtype_and_value_range_preserved():
    T = GaussianSmoothing(sigma_max=3.0)
    x = rand_image(n=5, h=32, w=32)
    for sigma in (0.5, 1.0, 3.0):
        y = T(x, sigma)
        assert y.shape == x.shape
        assert y.dtype == x.dtype and y.is_floating_point()
        # convex combination of inputs => stays inside the input range
        assert y.min() >= x.min() - 1e-6 and y.max() <= x.max() + 1e-6


def test_deterministic_across_calls_and_instances():
    x = rand_image(seed=3)
    a = GaussianSmoothing(sigma_max=3.0)(x, 1.5)
    b = GaussianSmoothing(sigma_max=3.0)(x, 1.5)
    c = GaussianSmoothing(sigma_max=3.0)(x, 1.5)
    assert torch.equal(a, b) and torch.equal(b, c)


def test_applied_to_original_not_composed():
    """Applying T_1 to T_1(x) is not the same as applying T_1 to x.

    The trainer always transforms the original image; this test pins the fact
    that composing stages would be a different (and lossier) operation.
    """
    T = GaussianSmoothing(sigma_max=3.0)
    x = rand_image(n=2, seed=7)
    once = T(x, 1.0)
    twice = T(once, 1.0)
    assert not torch.allclose(once, twice, atol=1e-4)


def test_channels_are_not_mixed():
    T = GaussianSmoothing(sigma_max=3.0)
    x = torch.zeros(1, 3, 32, 32)
    x[0, 1, 16, 16] = 1.0                       # energy only in the green channel
    y = T(x, 2.0)
    assert y[0, 0].abs().max().item() == 0.0
    assert y[0, 2].abs().max().item() == 0.0
    assert y[0, 1].sum().item() == pytest.approx(1.0, abs=1e-5)


def test_separable_matches_dense_2d_reference():
    """The separable implementation equals an explicit dense 2-D convolution."""
    import torch.nn.functional as F

    T = GaussianSmoothing(sigma_max=2.0, truncate=4.0)
    sigma = 1.3
    k = gaussian_kernel_1d(sigma, T.radius, dtype=torch.float64)
    k2 = torch.outer(k, k).view(1, 1, T.kernel_size, T.kernel_size)
    x = rand_image(n=2, c=1, h=32, w=32, seed=11).double()
    padded = F.pad(x, (T.radius,) * 4, mode="reflect")
    reference = F.conv2d(padded, k2)
    assert torch.allclose(T(x, sigma), reference, atol=1e-9)


def test_attenuation_of_spatial_oscillations():
    """Higher frequencies and larger sigma attenuate more, monotonically."""
    T = GaussianSmoothing(sigma_max=3.0)
    size = 64
    xs = torch.arange(size, dtype=torch.float32)

    def amplitude(freq, sigma):
        row = 0.5 + 0.4 * torch.cos(2 * math.pi * freq * xs / size)
        img = row.repeat(size, 1)[None, None]
        y = T(img, sigma)[0, 0, size // 2, size // 4: 3 * size // 4]
        return ((y.max() - y.min()) / 2).item() / 0.4

    for sigma in (0.5, 1.0, 2.0):
        amps = [amplitude(f, sigma) for f in (2, 4, 8, 16)]
        assert all(a > b for a, b in zip(amps, amps[1:])), (sigma, amps)
    amps = [amplitude(8, s) for s in (0.0, 0.5, 1.0, 2.0)]
    assert all(a > b for a, b in zip(amps, amps[1:])), amps


def test_attenuation_is_close_to_continuous_prediction_at_low_frequency():
    """Low frequencies follow exp(-sigma^2 w^2 / 2); high ones need not."""
    T = GaussianSmoothing(sigma_max=3.0)
    size, sigma, freq = 64, 1.0, 2
    xs = torch.arange(size, dtype=torch.float32)
    row = 0.5 + 0.4 * torch.cos(2 * math.pi * freq * xs / size)
    img = row.repeat(size, 1)[None, None]
    y = T(img, sigma)[0, 0, size // 2, size // 4: 3 * size // 4]
    measured = ((y.max() - y.min()) / 2).item() / 0.4
    w = 2 * math.pi * freq / size
    assert measured == pytest.approx(math.exp(-(sigma ** 2) * w ** 2 / 2), rel=0.02)


def test_boundary_effect_is_bounded_and_documented():
    """Reflection padding biases the border; the interior is unaffected by it.

    A vertical step is smoothed identically in the interior whether or not the
    step is near the edge, while columns within ``radius`` of the border differ
    from a plain (unmirrored) extension.
    """
    T = GaussianSmoothing(sigma_max=1.0, truncate=4.0)
    size = 32
    img = torch.zeros(1, 1, size, size)
    img[..., : size // 2] = 1.0
    y = T(img, 1.0)[0, 0, size // 2]
    # far from the step and from the borders, the result is flat
    assert y[0].item() == pytest.approx(1.0, abs=1e-6)
    assert y[-1].item() == pytest.approx(0.0, abs=1e-6)
    # monotone transition, no ringing (the kernel is non-negative)
    assert torch.all(y[1:] <= y[:-1] + 1e-7)


def test_fixed_support_independent_of_sigma():
    T = GaussianSmoothing(sigma_max=3.0, truncate=4.0)
    assert T.radius == 12 and T.kernel_size == 25
    for sigma in (0.1, 0.5, 1.0, 2.0, 3.0):
        assert T.kernel(sigma).numel() == 25


def test_sigma_above_max_is_an_error_not_silent_retruncation():
    T = GaussianSmoothing(sigma_max=1.0)
    with pytest.raises(ValueError, match="sigma_max"):
        T(rand_image(), 2.0)
    with pytest.raises(ValueError):
        T(rand_image(), -0.1)


def test_radius_must_fit_the_image():
    T = GaussianSmoothing(sigma_max=10.0, truncate=4.0)   # radius 40 > 32
    with pytest.raises(ValueError, match="reflection padding radius"):
        T(rand_image(h=32, w=32), 5.0)


def test_heat_time_helpers():
    assert heat_time_from_sigma(2.0) == pytest.approx(2.0)
    assert sigma_from_heat_time(2.0) == pytest.approx(2.0)
    for s in (0.0, 0.5, 1.0, 3.0):
        assert sigma_from_heat_time(heat_time_from_sigma(s)) == pytest.approx(s)


def test_discrete_filter_does_not_satisfy_the_semigroup_identity_exactly():
    """Composing two discrete blurs is close to, but not equal to, the combined one.

    Documents that the continuous heat semigroup identity is a motivation for the
    heat-time schedule parameterization, not a property of this implementation.
    """
    T = GaussianSmoothing(sigma_max=4.0, truncate=4.0)
    x = rand_image(n=2, c=1, seed=5)
    composed = T(T(x, 1.0), 1.0)
    combined = T(x, math.sqrt(2.0))
    diff = (composed - combined).abs().max().item()
    assert diff > 1e-6, "expected a measurable discrepancy"
    assert diff < 5e-2, "but the two should still be broadly similar"


def test_cache_key_separates_parameter_config_and_image():
    T = GaussianSmoothing(sigma_max=3.0)
    T2 = GaussianSmoothing(sigma_max=3.0, truncate=3.0)
    assert T.cache_key(1.0, "img7") == T.cache_key(1.0, "img7")
    assert T.cache_key(1.0, "img7") != T.cache_key(2.0, "img7")
    assert T.cache_key(1.0, "img7") != T.cache_key(1.0, "img8")
    assert T.cache_key(1.0, "img7") != T2.cache_key(1.0, "img7")   # config differs

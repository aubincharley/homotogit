"""Focused verification of the undecimated wavelet-shrinkage operator."""
import numpy as np
import pytest
import torch

from continuation.config import TransformConfig
from continuation.transforms import build_transform
from continuation.transforms.wavelet import (
    WAVELETS,
    band_rms,
    crop,
    get_filters,
    mirror_extend,
    soft_threshold,
    swt2_analysis,
    swt2_synthesis,
    wavelet_shrink,
)

ALL = list(WAVELETS)


def rand(b=2, c=3, h=16, w=16, seed=0, dtype=torch.float64):
    return torch.rand(b, c, h, w, generator=torch.Generator().manual_seed(seed),
                      dtype=dtype)


# ------------------------------------------------------- frame / round-trip

@pytest.mark.parametrize("wv", ALL)
def test_analysis_synthesis_roundtrip_is_exact(wv):
    """``W* W = I`` on the extended domain, exercised directly.

    This does not go through :func:`wavelet_shrink`, so it cannot be satisfied
    by any identity shortcut.
    """
    x = rand()
    a, det = swt2_analysis(x, wv, 2)
    rec = swt2_synthesis(a, det, wv)
    assert torch.allclose(rec, x, atol=1e-12), (wv, (rec - x).abs().max())


@pytest.mark.parametrize("wv", ALL)
def test_adjoint_identity(wv):
    """``<W u, v> == <u, W* v>`` for random ``u`` and random coefficient sets."""
    u = rand(seed=1)
    a_u, d_u = swt2_analysis(u, wv, 2)
    a_v, d_v = swt2_analysis(rand(seed=2), wv, 2)      # any valid coefficient shape
    v_syn = swt2_synthesis(a_v, d_v, wv)
    lhs = float((a_u * a_v).sum()) + sum(
        float((d_u[j][o] * d_v[j][o]).sum()) for j in range(2) for o in ("LH", "HL", "HH"))
    rhs = float((u * v_syn).sum())
    assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("wv", ALL)
def test_parseval_energy(wv):
    x = rand(seed=3)
    a, det = swt2_analysis(x, wv, 2)
    total = float((a ** 2).sum()) + sum(
        float((det[j][o] ** 2).sum()) for j in range(2) for o in ("LH", "HL", "HH"))
    assert total == pytest.approx(float((x ** 2).sum()), rel=1e-12)


@pytest.mark.parametrize("wv", ALL)
def test_band_energies_match_pywavelets(wv):
    """Independent reference: band energies agree with ``pywt.swt2(norm=True)``.

    Energies are compared as a multiset because our alignment differs from
    PyWavelets by a per-band circular shift (a shift of a Parseval frame is
    still the same frame); orientation labels may also be permuted.
    """
    pywt = pytest.importorskip("pywt")
    x = rand(b=1, c=1, seed=4)
    a, det = swt2_analysis(x, wv, 2)
    mine = sorted([float((a ** 2).sum())]
                  + [float((det[j][o] ** 2).sum())
                     for j in range(2) for o in ("LH", "HL", "HH")])
    ref = pywt.swt2(x[0, 0].numpy(), wv, level=2, norm=True)
    cA2, coarse = ref[0]
    fine = ref[1][1]
    theirs = sorted([float((cA2 ** 2).sum())]
                    + [float((b ** 2).sum()) for b in coarse]
                    + [float((b ** 2).sum()) for b in fine])
    assert np.allclose(mine, theirs, atol=1e-10)


def test_filters_are_normalized_for_a_tight_frame():
    for wv in ALL:
        lo, hi = get_filters(wv, torch.float64)
        # |H|^2 + |G|^2 = 1 after the 1/sqrt(2) scaling
        assert float((lo ** 2).sum() + (hi ** 2).sum()) == pytest.approx(1.0, rel=1e-12)


def test_non_orthogonal_wavelet_is_rejected():
    with pytest.raises(ValueError, match="orthogonal"):
        get_filters("bior2.2", torch.float64)


# ----------------------------------------------------------- boundary / P

def test_mirror_extension_and_crop():
    x = rand(b=1, c=1, h=4, w=4)
    e = mirror_extend(x)
    assert e.shape[-2:] == (8, 8)
    assert torch.equal(crop(e, 4, 4), x)
    assert torch.equal(e[..., :4, 4:], torch.flip(x, dims=[-1]))
    assert torch.equal(e[..., 4:, :4], torch.flip(x, dims=[-2]))


# ------------------------------------------------------------- shrinkage

@pytest.mark.parametrize("wv", ALL)
def test_identity_at_s_equals_one_without_the_bypass(wv):
    """``T_1 = I`` produced by the actual round-trip, bypass disabled."""
    x = rand(seed=5)
    for impl in ("reference", "fast"):
        y = wavelet_shrink(x, 1.0, wv, impl=impl, bypass_identity=False)
        assert torch.allclose(y, x, atol=1e-11), (wv, impl, (y - x).abs().max())


@pytest.mark.parametrize("wv", ALL)
def test_identity_bypass_returns_the_input_exactly(wv):
    x = rand(seed=5)
    assert wavelet_shrink(x, 1.0, wv) is x
    y, info = wavelet_shrink(x, 1.0, wv, return_info=True)
    assert y is x and info["identity_bypass"] is True


@pytest.mark.parametrize("wv", ALL)
def test_fast_matches_reference_values_and_gradients(wv):
    """The optimized path must be numerically indistinguishable, for every wavelet."""
    for s in (0.0, 0.25, 0.5, 0.9):
        x = rand(seed=13)
        a = x.clone().requires_grad_(True)
        b = x.clone().requires_grad_(True)
        ya = wavelet_shrink(a, s, wv, impl="reference")
        yb = wavelet_shrink(b, s, wv, impl="fast")
        assert torch.allclose(ya, yb, atol=1e-12), (wv, s, (ya - yb).abs().max())
        (ya * ya).sum().backward()
        (yb * yb).sum().backward()
        assert torch.allclose(a.grad, b.grad, atol=1e-11), (wv, s)


@pytest.mark.parametrize("wv", ALL)
def test_checkpointing_is_exact_not_straight_through(wv):
    x = rand(seed=14)
    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    ya = wavelet_shrink(a, 0.5, wv, checkpoint=False)
    yb = wavelet_shrink(b, 0.5, wv, checkpoint=True)
    assert torch.allclose(ya, yb, atol=1e-12)
    (ya * ya).sum().backward()
    (yb * yb).sum().backward()
    assert torch.allclose(a.grad, b.grad, atol=1e-11)


def test_gradcheck_fast_path():
    x = rand(b=1, c=1, h=8, w=8, seed=15).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda t: wavelet_shrink(t, 0.5, "db2", impl="fast").sum(), (x,),
        eps=1e-6, atol=1e-6)


def test_impl_must_be_known():
    with pytest.raises(ValueError, match="impl"):
        wavelet_shrink(rand(), 0.5, "haar", impl="turbo")


@pytest.mark.parametrize("wv", ALL)
def test_constant_input_is_preserved_at_every_s(wv):
    """Details of a constant image vanish, so shrinkage cannot change it."""
    x = torch.full((2, 3, 16, 16), 0.37, dtype=torch.float64)
    _, det = swt2_analysis(mirror_extend(x), wv, 2)
    for j in range(2):
        for o in ("LH", "HL", "HH"):
            assert det[j][o].abs().max() < 1e-12
    for s in (0.0, 0.25, 0.5, 1.0):
        assert torch.allclose(wavelet_shrink(x, s, wv), x, atol=1e-11), (wv, s)


@pytest.mark.parametrize("wv", ALL)
def test_shrinkage_is_monotone_in_s(wv):
    """Smaller ``s`` means larger thresholds, so more detail is removed."""
    x = rand(seed=6)
    errs = [float((wavelet_shrink(x, s, wv, bypass_identity=False) - x).norm())
            for s in (1.0, 0.75, 0.5, 0.25, 0.0)]
    assert all(a <= b + 1e-12 for a, b in zip(errs, errs[1:])), (wv, errs)
    assert errs[0] < 1e-10 and errs[-1] > 0


def test_zero_fraction_increases_as_s_decreases():
    x = rand(seed=7)
    prev = -1.0
    for s in (1.0, 0.75, 0.5, 0.25, 0.0):
        _, info = wavelet_shrink(x, s, "db2", return_info=True)
        frac = float(torch.stack(list(info["zero_fraction"].values())).mean())
        assert frac >= prev - 1e-9
        prev = frac
    assert prev > 0.5


def test_rms_is_per_sample_and_channel_not_pooled():
    """Scaling one sample must not change another sample's threshold."""
    x = rand(b=2, c=2, seed=8)
    y_ref = wavelet_shrink(x, 0.5, "db2")
    x2 = x.clone()
    x2[1] *= 10.0
    y2 = wavelet_shrink(x2, 0.5, "db2")
    assert torch.allclose(y2[0], y_ref[0], atol=1e-12)
    # and the rescaled sample shrinks proportionally (thresholds scale with RMS)
    assert torch.allclose(y2[1], 10.0 * y_ref[1], atol=1e-9)


def test_soft_threshold_and_band_rms():
    v = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0], dtype=torch.float64)
    out = soft_threshold(v, torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(out, torch.tensor([-2.0, 0.0, 0.0, 0.0, 2.0], dtype=torch.float64))
    d = torch.full((1, 1, 4, 4), 2.0, dtype=torch.float64)
    assert float(band_rms(d)) == pytest.approx(2.0, rel=1e-9)


def test_zero_band_gives_finite_zero_gradient():
    """An all-zero detail band must not produce an infinite RMS gradient."""
    d = torch.zeros(1, 1, 8, 8, dtype=torch.float64, requires_grad=True)
    band_rms(d).sum().backward()
    assert torch.isfinite(d.grad).all() and float(d.grad.abs().max()) == 0.0


# ------------------------------------------------------------- autograd

@pytest.mark.parametrize("wv", ["haar", "db2"])
def test_gradcheck_away_from_threshold_boundaries(wv):
    x = rand(b=1, c=1, h=8, w=8, seed=9).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda t: wavelet_shrink(t, 0.5, wv).sum(), (x,), eps=1e-6, atol=1e-6)


def test_gradients_flow_through_the_rms_thresholds():
    x = rand(b=1, c=1, h=8, w=8, seed=10).requires_grad_(True)
    wavelet_shrink(x, 0.5, "db2").pow(2).sum().backward()
    assert torch.isfinite(x.grad).all() and float(x.grad.abs().max()) > 0


# ------------------------------------------------------------- interface

def test_transform_interface_and_range_reporting():
    T = build_transform(TransformConfig(family="wavelet", params={"wavelet": "sym4"}))
    assert T.target_parameter == 1.0 and T.is_target(1.0)
    assert T.parameter_name == "s"
    x = torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(11))
    out = T.apply(x, 0.5)
    assert out.images.shape == x.shape and out.images.dtype == x.dtype
    assert "out_of_unit_range" in out.info and "zero_fraction" in out.info
    assert torch.allclose(T.apply(x, 1.0).images, x, atol=1e-5)
    with pytest.raises(ValueError):
        T.apply(x, 1.5)


def test_cache_key_separates_wavelet_and_parameter():
    a = build_transform(TransformConfig(family="wavelet", params={"wavelet": "db2"}))
    b = build_transform(TransformConfig(family="wavelet", params={"wavelet": "sym4"}))
    assert a.cache_key(0.5, "img1") != b.cache_key(0.5, "img1")
    assert a.cache_key(0.5, "img1") != a.cache_key(0.25, "img1")


def test_filter_support_must_fit_the_extended_axis():
    """sym4 at level 2 needs 15 taps of support; a 4x4 input extends to 8x8."""
    x = torch.rand(1, 1, 4, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="does not fit"):
        wavelet_shrink(x, 0.5, "sym4")


def test_works_on_non_image_channel_counts():
    """The operator is defined for activations, not just 3-channel images."""
    x = torch.rand(2, 16, 8, 8, dtype=torch.float64)
    y = wavelet_shrink(x, 0.5, "haar")
    assert y.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_matches_cpu():
    x = rand(b=1, c=3, h=16, w=16, seed=12, dtype=torch.float32)
    a = wavelet_shrink(x, 0.5, "db2")
    b = wavelet_shrink(x.cuda(), 0.5, "db2").cpu()
    assert torch.allclose(a, b, atol=1e-5)

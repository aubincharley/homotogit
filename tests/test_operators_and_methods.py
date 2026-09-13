import math

import pytest
import torch

from continuation_core.controller import InterventionController, InterventionState
from continuation_core.methods import METHODS, PLATEAU_G, RPROG, MethodSpec, get_method
from continuation_core.models import site_map
from continuation_core.operators import (FixedSupportGaussian, adaptive_max_reduce,
                                         adaptive_windows, reflected_indices)
from continuation_core.schedules import EpochSchedule


def test_gaussian_support_identity_and_limits():
    g = FixedSupportGaussian(1.0, 4.0)
    assert (g.radius, g.kernel_size) == (4, 9)
    x = torch.rand(2, 3, 8, 8)
    assert g(x, 0.0) is x
    c = torch.full((1, 2, 4, 4), 0.3)
    assert torch.allclose(g(c, 0.7), c, atol=1e-6)
    with pytest.raises(ValueError):
        g(x, 1.5)
    k = g.kernel(0.5, torch.float32, None)
    assert abs(float(k.sum()) - 1.0) < 1e-6 and k.numel() == 9


def test_explicit_reflection_on_small_maps():
    assert reflected_indices(4, 4).tolist() == [2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1]
    y = FixedSupportGaussian(1.0)(torch.rand(1, 1, 4, 4), 0.5)
    assert y.shape == (1, 1, 4, 4) and torch.isfinite(y).all()


def test_adaptive_max_windows_and_bypass():
    w24 = adaptive_windows(32, 24)
    overlaps = sum(1 for j in range(1, 24) if w24[j][0] < w24[j - 1][1])
    assert overlaps == 16
    assert all(b - a == 2 for a, b in adaptive_windows(32, 16))
    x = torch.rand(1, 2, 32, 32)
    assert adaptive_max_reduce(x, 32) is x and adaptive_max_reduce(x, None) is x
    assert adaptive_max_reduce(x, 24).shape[-1] == 24


def test_schedules_are_the_executed_ones():
    assert PLATEAU_G.table(30) == [1.0] * 3 + [0.85] * 3 + [0.7] * 3 + [0.6] * 3 + \
        [0.5] * 3 + [0.4] * 3 + [0.3] * 3 + [0.0] * 9
    assert RPROG.table(30) == [16] * 6 + [24] * 6 + [32] * 18
    assert RPROG.transitions() == (6, 12)
    assert PLATEAU_G.at(45) == 0.0
    with pytest.raises(ValueError):
        EpochSchedule((1, 2), (0, 0))


def test_combined_method_effective_sigma_table():
    c = InterventionController(get_method("resolution_max_b1_gaussian_conv"),
                               site_map("resnet20_bn_cifar"))
    expected = {0: 0.5, 3: 0.425, 6: 0.75 * 0.7, 9: 0.75 * 0.6, 12: 0.5, 15: 0.4,
                18: 0.3, 21: 0.0}
    for e, s in expected.items():
        c.set_epoch(e)
        sig = c.per_site_sigma()
        assert len(sig) == 19 and set(sig) == {s}


def test_gaussian_postrelu_sigma_equals_g_at_ten_sites():
    c = InterventionController(get_method("gaussian_postrelu"), site_map("resnet20_bn_cifar"))
    for e in range(30):
        c.set_epoch(e)
        assert c.state.resolution is None
        assert c.per_site_sigma() == [PLATEAU_G.at(e) if PLATEAU_G.at(e) > 0 else 0.0] * 10


def test_method_spec_roundtrip_and_non_factorial():
    for m in METHODS.values():
        assert MethodSpec.from_dict(m.to_dict()) == m
    a, b = METHODS["gaussian_postrelu"], METHODS["resolution_max_b1_gaussian_conv"]
    assert a.gaussian.placement != b.gaussian.placement
    assert a.gaussian.sigma_scale != b.gaussian.sigma_scale
    with pytest.raises(KeyError):
        get_method("maxblur")


def test_state_serialisation():
    s = InterventionState(24, 0.7, "epoch 6")
    assert InterventionState.from_dict(s.to_dict()) == s

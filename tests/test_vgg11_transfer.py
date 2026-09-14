"""VGG-11-BN: the non-residual architecture, and its spatial headroom.

The architecture-specific risks here are different from the dataset transfers.
VGG-11 has five max pools, so a 32x32 image is already reduced to 1x1 by the end;
any resolution intervention pushes a pool onto a 1x1 map, which with the default
floor rounding is 0x0 and raises. ``ceil_mode=True`` fixes that, and must be
provably invisible to the network as published.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from continuation_core.controller import (InterventionController,           # noqa: E402
                                          UnsupportedInsertionError)
from continuation_core.methods import PLATEAU_G, get_method                 # noqa: E402
from continuation_core.models import build_model, is_validated, site_map    # noqa: E402
from continuation_core.presets import CIFAR10_MEAN                          # noqa: E402

import vgg11_configs as C                                                   # noqa: E402

ARCH = "vgg11_bn"


def _model_and_controller(method_id):
    torch.manual_seed(0)
    m = build_model(ARCH, 10)
    cfg = C.build(method_id, 0, data_root="d", assets_dir="a", out_dir="runs")
    c = InterventionController(cfg.method_spec(), site_map(ARCH), ARCH)
    c.attach(m)
    return m, c, cfg


# -- ceil_mode must not change the published network ---------------------------

def test_ceil_mode_leaves_the_plain_network_bitwise_identical():
    torch.manual_seed(0)
    a = build_model(ARCH, 10)
    torch.manual_seed(0)
    b = build_model(ARCH, 10)
    for m in b.modules():                       # back to the published rounding
        if isinstance(m, nn.MaxPool2d):
            m.ceil_mode = False
    b.load_state_dict(a.state_dict())
    a.eval(), b.eval()
    x = torch.rand(8, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(a(x), b(x))


def test_every_native_pool_input_is_even_which_is_why():
    """ceil and floor agree only because no native size is odd."""
    m = build_model(ARCH, 10)
    seen = []
    for mod in m.features:
        if isinstance(mod, nn.MaxPool2d):
            mod.register_forward_pre_hook(
                lambda _m, inp: seen.append(int(inp[0].shape[-1])))
    m(torch.rand(1, 3, 32, 32))
    assert seen == [32, 16, 8, 4, 2]
    assert all(s % 2 == 0 for s in seen)


def test_floor_rounding_would_crash_under_the_reduction():
    """The reason ceil_mode is there at all."""
    m, c, _ = _model_and_controller("resolution_max_b1")
    for mod in m.modules():
        if isinstance(mod, nn.MaxPool2d):
            mod.ceil_mode = False
    c.set_epoch(0)                                   # r = 8
    with pytest.raises(RuntimeError, match="too small"):
        m(torch.rand(2, 3, 32, 32))


# -- the site map --------------------------------------------------------------

def test_site_counts_differ_from_resnet20_and_that_is_expected():
    sm = site_map(ARCH)
    assert len(sm["conv_out"]) == 8            # 19 on ResNet-20
    assert len(sm["post_relu"]) == 8           # 10 on ResNet-20
    assert not is_validated(ARCH)


def test_block1_is_still_refused():
    """A ResNet-20 name must not be silently redirected to the new point."""
    with pytest.raises(UnsupportedInsertionError, match="block1"):
        InterventionController(get_method("resolution_max_b1"), site_map(ARCH), ARCH)


def test_after_pool1_is_the_input_of_features_4():
    assert site_map(ARCH)["reduction"] == {"after_pool1": ("input", "features.4")}


# -- schedules and grids -------------------------------------------------------

def test_gaussian_levels_are_the_reference_unchanged():
    assert C.VGG_G is PLATEAU_G
    assert C.EPOCHS == 30 and C.NATIVE_RESOLUTION == 16


def test_resolution_keeps_the_reference_ratios_against_a_native_16():
    assert C.VGG_R.values == (8, 12, 16)
    for vgg, ref in zip(C.VGG_R.values, (16, 24, 32)):
        assert vgg / C.NATIVE_RESOLUTION == ref / 32


@pytest.mark.parametrize("epoch,r,expected", [
    (0, 8, {4: 8, 8: 4, 15: 2, 22: 1, 28: 1}),
    (6, 12, {4: 12, 8: 6, 15: 3, 22: 2, 28: 1}),
    (12, 16, {4: 16, 8: 8, 15: 4, 22: 2, 28: 1}),
])
def test_grids_under_each_state(epoch, r, expected):
    m, c, _ = _model_and_controller("resolution_max_b1")
    seen = {}
    for i in expected:
        m.features[i].register_forward_hook(
            lambda _mo, _i, o, i=i: seen.__setitem__(i, int(o.shape[-1])))
    c.set_epoch(epoch)
    assert c.state.resolution == r
    assert m(torch.rand(2, 3, 32, 32)).shape == (2, 10)
    assert seen == expected


# -- the invariant the whole benchmark rests on --------------------------------

@pytest.mark.parametrize("method_id", C.METHODS)
def test_target_state_is_bitwise_plain(method_id):
    m, c, _ = _model_and_controller(method_id)
    torch.manual_seed(0)
    plain = build_model(ARCH, 10)
    plain.load_state_dict(m.state_dict())
    c.set_state(c.target_state())
    m.eval(), plain.eval()
    x = torch.rand(4, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(m(x), plain(x))


@pytest.mark.parametrize("method_id", C.METHODS)
def test_forward_and_backward_are_finite_at_every_state(method_id):
    import torch.nn.functional as F
    m, c, cfg = _model_and_controller(method_id)
    x, y = torch.rand(4, 3, 32, 32), torch.randint(0, 10, (4,))
    seen = set()
    for e in range(cfg.budget.epochs):
        st = c.state_for_epoch(e)
        if (st.resolution, st.sigma) in seen:
            continue
        seen.add((st.resolution, st.sigma))
        c.set_state(st)
        m.zero_grad(set_to_none=True)
        out = m(x)
        F.cross_entropy(out, y).backward()
        assert torch.isfinite(out).all()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


# -- configs -------------------------------------------------------------------

@pytest.mark.parametrize("method_id", C.METHODS)
def test_config_is_the_reference_recipe_on_a_new_architecture(method_id, tmp_path):
    cfg = C.build(method_id, 0, data_root="d", assets_dir="a", out_dir="runs")
    assert cfg.model.arch == ARCH
    assert cfg.validation_status == "unvalidated"
    assert cfg.budget.epochs == 30 and cfg.optimizer.lr == 0.005
    assert cfg.data.name == "cifar10" and cfg.data.expected_mean == CIFAR10_MEAN
    spec = cfg.method_spec()
    if spec.resolution:
        assert spec.resolution.point == "after_pool1"
        assert spec.resolution.reference_resolution == 16
    if spec.gaussian:
        assert spec.gaussian.schedule.values == PLATEAU_G.values
    p = tmp_path / "c.json"
    cfg.save(p)
    assert type(cfg).load(p).to_dict() == cfg.to_dict()

"""ResNet-20 with GroupNorm: the BatchNorm control.

The experiment only means something if the swap is *only* the normalisation: the
same parameters, the same insertion sites, the same schedules, the same data.
These tests pin that, and pin the property the control depends on -- that
GroupNorm has no batch-fitted state for the Gaussian to corrupt.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from continuation_core.controller import InterventionController         # noqa: E402
from continuation_core.data import InputPipeline                        # noqa: E402
from continuation_core.evaluate import evaluate                         # noqa: E402
from continuation_core.methods import PLATEAU_G, RPROG                  # noqa: E402
from continuation_core.models import (available, build_model,           # noqa: E402
                                      is_validated, site_map)
from continuation_core.presets import CIFAR10_MEAN                      # noqa: E402

import resnet20_gn_configs as G                                         # noqa: E402

GN, BN = "resnet20_gn_cifar", "resnet20_bn_cifar"


def _pair(method_id, arch=GN):
    torch.manual_seed(0)
    m = build_model(arch, 10, **({"groups": G.GROUPS} if arch == GN else {}))
    cfg = G.build(method_id, 0, data_root="d", assets_dir="a", out_dir="runs")
    c = InterventionController(cfg.method_spec(), site_map(arch), arch)
    c.attach(m)
    return m, c, cfg


# -- only the normalisation differs -------------------------------------------

def test_registered_and_unvalidated():
    assert GN in available() and not is_validated(GN)


def test_same_parameter_count_as_batchnorm():
    a, b = build_model(BN, 10), build_model(GN, 10)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_groupnorm_has_no_batch_fitted_state():
    """The whole point: nothing for the blur to corrupt, nothing to re-estimate."""
    gn = build_model(GN, 10)
    assert list(gn.buffers()) == []
    assert len(list(build_model(BN, 10).buffers())) > 0
    assert not any(isinstance(m, nn.BatchNorm2d) for m in gn.modules())
    assert any(isinstance(m, nn.GroupNorm) for m in gn.modules())


def test_site_map_is_the_batchnorm_one_verbatim():
    """It names convolutions and blocks, never a normalisation layer, so the
    methods attach at identical tensors in both models."""
    assert site_map(GN) == site_map(BN)
    gn = build_model(GN, 10)
    for name in site_map(GN)["conv_out"]:
        assert isinstance(gn.get_submodule(name), nn.Conv2d)
    for _, name in site_map(GN)["post_relu"]:
        gn.get_submodule(name)
    gn.get_submodule(site_map(GN)["reduction"]["block1"][1])


def test_bn_policy_is_inert_for_groupnorm_but_not_for_batchnorm():
    torch.manual_seed(0)
    pipe = InputPipeline(torch.zeros(3), torch.ones(3))
    x = torch.randint(0, 256, (100, 3, 32, 32), dtype=torch.uint8)
    y = torch.randint(0, 10, (100,))
    out = {}
    for arch in (GN, BN):
        m, c, _ = _pair("plain", arch)
        out[arch] = [evaluate(m, c, pipe, x, y, c.target_state(), bn_policy=p,
                              batch_size=25)["ce"]
                     for p in ("running_stats", "fixed_batch_stats")]
    assert out[GN][0] == out[GN][1]        # no batch dependence at all
    assert out[BN][0] != out[BN][1]        # BatchNorm does depend on it


@pytest.mark.parametrize("groups", [3, 5, 24])
def test_groups_must_divide_every_width(groups):
    with pytest.raises(ValueError, match="does not divide"):
        build_model(GN, 10, groups=groups)


def test_default_groups_divide_all_three_widths():
    assert all(w % G.GROUPS == 0 for w in (16, 32, 64))
    build_model(GN, 10, groups=G.GROUPS)


def test_unknown_option_is_rejected():
    with pytest.raises(TypeError, match="unknown options"):
        build_model(GN, 10, momentum=0.1)


# -- the invariants every study in this benchmark rests on ---------------------

@pytest.mark.parametrize("method_id", G.METHODS)
def test_target_state_is_bitwise_plain(method_id):
    m, c, _ = _pair(method_id)
    torch.manual_seed(0)
    plain = build_model(GN, 10, groups=G.GROUPS)
    plain.load_state_dict(m.state_dict())
    c.set_state(c.target_state())
    m.eval(), plain.eval()
    x = torch.rand(4, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(m(x), plain(x))


@pytest.mark.parametrize("method_id", G.METHODS)
def test_forward_and_backward_finite_at_every_state(method_id):
    m, c, cfg = _pair(method_id)
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


def test_reduction_changes_every_downstream_grid():
    m, c, _ = _pair("resolution_max_b1")
    seen = {}
    for i in (1, 2, 3, 6, 8):
        m.blocks[i].register_forward_hook(
            lambda _mo, _i, o, i=i: seen.__setitem__(i, int(o.shape[-1])))
    c.set_epoch(0)
    m(torch.rand(2, 3, 32, 32))
    assert seen == {1: 32, 2: 16, 3: 8, 6: 4, 8: 4}


# -- configs -------------------------------------------------------------------

@pytest.mark.parametrize("method_id", G.METHODS)
def test_config_is_the_reference_recipe_with_one_change(method_id, tmp_path):
    cfg = G.build(method_id, 0, data_root="d", assets_dir="a", out_dir="runs")
    assert cfg.model.arch == GN and cfg.model.options == {"groups": G.GROUPS}
    assert cfg.validation_status == "unvalidated"
    assert cfg.budget.epochs == 30 and cfg.optimizer.lr == 0.005
    assert cfg.data.name == "cifar10" and cfg.data.expected_mean == CIFAR10_MEAN
    spec = cfg.method_spec()
    if spec.gaussian:
        assert spec.gaussian.schedule.values == PLATEAU_G.values
        assert spec.gaussian.sigma_max == 1.0
    if spec.resolution:
        assert spec.resolution.point == "block1"            # unchanged from reference
        assert spec.resolution.schedule.values == RPROG.values
        assert spec.resolution.reference_resolution == 32
    p = tmp_path / "c.json"
    cfg.save(p)
    assert type(cfg).load(p).to_dict() == cfg.to_dict()


def test_schedules_are_the_reference_verbatim():
    assert G.GN_G is PLATEAU_G and G.GN_R is RPROG

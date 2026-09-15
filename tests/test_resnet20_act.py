"""ResNet-20 with GELU / SiLU: the ReLU control.

The claim this study rests on is that the *only* difference from the reference is
the activation function -- same initial weights, same data order, same sites. That
is only true if the state dict is interchangeable and the relu arm reproduces
resnet20_bn exactly, so both are asserted here rather than assumed.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from continuation_core.controller import InterventionController         # noqa: E402
from continuation_core.methods import PLATEAU_G, RPROG                  # noqa: E402
from continuation_core.models import (available, build_model,           # noqa: E402
                                      is_validated, site_map)
import resnet20_act_configs as A                                        # noqa: E402

ACT, BN = "resnet20_act_cifar", "resnet20_bn_cifar"
REFERENCE_INIT = ROOT / "assets" / "cifar10_resnet20bn" / "init_seed0.pt"


def _pair(method_id, activation="gelu"):
    torch.manual_seed(0)
    m = build_model(ACT, 10, activation=activation)
    cfg = A.build(method_id, 0, activation, data_root="d", assets_dir="a", out_dir="runs")
    c = InterventionController(cfg.method_spec(), site_map(ACT), ACT)
    c.attach(m)
    return m, c, cfg


# -- the port is exact ---------------------------------------------------------

def test_registered_and_unvalidated():
    assert ACT in available() and not is_validated(ACT)


def test_relu_arm_is_bitwise_resnet20_bn():
    """If this drifts, the whole comparison loses its control."""
    torch.manual_seed(0)
    bn = build_model(BN, 10)
    act = build_model(ACT, 10, activation="relu")
    act.load_state_dict(bn.state_dict(), strict=True)
    bn.eval(), act.eval()
    x = torch.rand(8, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(bn(x), act(x))


@pytest.mark.parametrize("activation", ["relu", "gelu", "silu"])
def test_state_dict_is_interchangeable_with_resnet20_bn(activation):
    bn = build_model(BN, 10)
    m = build_model(ACT, 10, activation=activation)
    assert list(m.state_dict()) == list(bn.state_dict())
    m.load_state_dict(bn.state_dict(), strict=True)


def test_the_pinned_reference_assets_load_directly():
    """Why this study needs no asset set of its own."""
    sd = torch.load(REFERENCE_INIT, map_location="cpu", weights_only=True)
    for activation in ("gelu", "silu"):
        build_model(ACT, 10, activation=activation).load_state_dict(sd, strict=True)
    assert A.REFERENCE_ASSETS == "assets/cifar10_resnet20bn"


@pytest.mark.parametrize("activation", ["gelu", "silu"])
def test_the_activation_actually_changes_the_output(activation):
    torch.manual_seed(0)
    bn = build_model(BN, 10)
    m = build_model(ACT, 10, activation=activation)
    m.load_state_dict(bn.state_dict())
    bn.eval(), m.eval()
    x = torch.rand(8, 3, 32, 32)
    with torch.no_grad():
        assert not torch.equal(bn(x), m(x))


def test_unknown_activation_is_rejected():
    with pytest.raises(ValueError, match="unknown activation"):
        build_model(ACT, 10, activation="mish")
    with pytest.raises(TypeError, match="unknown options"):
        build_model(ACT, 10, negative_slope=0.1)


def test_site_map_is_activation_agnostic():
    assert site_map(ACT) == site_map(BN)


@pytest.mark.parametrize("activation", ["gelu", "silu"])
def test_the_named_function_is_the_one_applied(activation):
    """post_relu sites hook block outputs, so the hooked tensor must carry the
    signature of the chosen activation -- negative values, unlike ReLU."""
    m = build_model(ACT, 10, activation=activation)
    seen = {}
    m.blocks[0].register_forward_hook(lambda _m, _i, o: seen.update(out=o))
    m.eval()
    with torch.no_grad():
        m(torch.rand(8, 3, 32, 32) * 4 - 2)
    assert float(seen["out"].min()) < 0.0           # ReLU could never do this
    relu = build_model(ACT, 10, activation="relu")
    relu.load_state_dict(m.state_dict())
    seen_r = {}
    relu.blocks[0].register_forward_hook(lambda _m, _i, o: seen_r.update(out=o))
    relu.eval()
    with torch.no_grad():
        relu(torch.rand(8, 3, 32, 32) * 4 - 2)
    assert float(seen_r["out"].min()) == 0.0


# -- the benchmark invariants --------------------------------------------------

@pytest.mark.parametrize("activation", A.ARMS)
@pytest.mark.parametrize("method_id", A.METHODS)
def test_target_state_is_bitwise_plain(method_id, activation):
    m, c, _ = _pair(method_id, activation)
    torch.manual_seed(0)
    plain = build_model(ACT, 10, activation=activation)
    plain.load_state_dict(m.state_dict())
    c.set_state(c.target_state())
    m.eval(), plain.eval()
    x = torch.rand(4, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(m(x), plain(x))


@pytest.mark.parametrize("method_id", A.METHODS)
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


# -- configs -------------------------------------------------------------------

@pytest.mark.parametrize("activation", A.ARMS)
@pytest.mark.parametrize("method_id", A.METHODS)
def test_config_is_the_reference_recipe_with_one_change(method_id, activation, tmp_path):
    cfg = A.build(method_id, 0, activation, data_root="d",
                  assets_dir=A.REFERENCE_ASSETS, out_dir="runs")
    assert cfg.model.arch == ACT and cfg.model.options == {"activation": activation}
    assert cfg.assets.dir == "assets/cifar10_resnet20bn"     # the reference's own
    assert cfg.budget.epochs == 30 and cfg.optimizer.lr == 0.005
    spec = cfg.method_spec()
    if spec.gaussian:
        assert spec.gaussian.schedule.values == PLATEAU_G.values
    if spec.resolution:
        assert spec.resolution.schedule.values == RPROG.values
        assert spec.resolution.point == "block1"
    p = tmp_path / "c.json"
    cfg.save(p)
    assert type(cfg).load(p).to_dict() == cfg.to_dict()


def test_relu_is_not_an_arm():
    """It would be resnet20_bn; the reference table is its control."""
    assert "relu" not in A.ARMS and set(A.ARMS) == {"gelu", "silu"}
    with pytest.raises(SystemExit):
        A.build("plain", 0, "relu", data_root="d", assets_dir="a", out_dir="r")

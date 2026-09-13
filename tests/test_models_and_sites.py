import pytest
import torch

from continuation_core.controller import InterventionController, UnsupportedInsertionError
from continuation_core.methods import METHODS, get_method
from continuation_core.models import build_model, site_map


def _pair(method_id, arch="resnet20_bn_cifar"):
    torch.manual_seed(0)
    m = build_model(arch, 10)
    c = InterventionController(get_method(method_id), site_map(arch), arch)
    return m, c, c.attach(m)


def test_resnet20_site_map_names_exist_and_counts():
    m = build_model("resnet20_bn_cifar", 10)
    sm = site_map("resnet20_bn_cifar")
    for name in sm["conv_out"]:
        assert isinstance(m.get_submodule(name), torch.nn.Conv2d)
    assert len(sm["conv_out"]) == 19 and len(sm["post_relu"]) == 10
    assert sum(p.numel() for p in m.parameters()) == 269722


def test_reduction_changes_every_downstream_grid():
    m, c, _ = _pair("resolution_max_b1")
    seen = {}
    for i in (1, 2, 3, 6, 8):
        m.blocks[i].register_forward_hook(lambda _m, _i, o, i=i: seen.__setitem__(i, o.shape[-1]))
    c.set_epoch(0)
    out = m(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 10)
    assert seen == {1: 32, 2: 16, 3: 8, 6: 4, 8: 4}
    c.set_epoch(6)
    m(torch.rand(2, 3, 32, 32))
    assert seen == {1: 32, 2: 24, 3: 12, 6: 6, 8: 6}


def test_post_relu_hooks_fire_exactly_ten_times():
    m, c, handles = _pair("gaussian_postrelu")
    assert len(handles) == 10
    calls = []
    orig = c._blur
    c._blur = lambda s, h: (calls.append(s), orig(s, h))[1]
    c.set_epoch(0)
    m(torch.rand(1, 3, 32, 32))
    assert calls == list(range(10))


def test_conv_out_hooks_fire_nineteen_times_before_bn():
    m, c, handles = _pair("resolution_max_b1_gaussian_conv")
    assert len(handles) == 20                       # 1 reduction + 19 conv outputs
    calls = []
    orig = c._blur
    c._blur = lambda s, h: (calls.append(s), orig(s, h))[1]
    c.set_epoch(0)
    m(torch.rand(1, 3, 32, 32))
    assert calls == list(range(19))


@pytest.mark.parametrize("method_id", sorted(METHODS))
def test_target_state_is_bitwise_plain(method_id):
    m, c, _ = _pair(method_id)
    torch.manual_seed(1)
    plain = build_model("resnet20_bn_cifar", 10)
    plain.load_state_dict(m.state_dict())
    x = torch.rand(4, 3, 32, 32)
    c.set_state(c.target_state())
    for mode in (True, False):
        m.train(mode), plain.train(mode)
        with torch.no_grad():
            assert torch.equal(m(x), plain(x))


def test_vgg_adapter_supports_gaussian_but_refuses_block1():
    m = build_model("vgg11_bn", 10, width=16)
    sm = site_map("vgg11_bn")
    assert len(sm["conv_out"]) == 8 and len(sm["post_relu"]) == 8
    for name in sm["conv_out"]:
        assert isinstance(m.get_submodule(name), torch.nn.Conv2d)
    for _, name in sm["post_relu"]:
        assert isinstance(m.get_submodule(name), torch.nn.ReLU)
    c = InterventionController(get_method("gaussian_postrelu"), sm, "vgg11_bn")
    c.attach(m)
    c.set_epoch(0)
    assert m(torch.rand(2, 3, 32, 32)).shape == (2, 10)
    for mid in ("resolution_max_b1", "resolution_max_b1_gaussian_conv"):
        with pytest.raises(UnsupportedInsertionError, match="block1"):
            InterventionController(get_method(mid), sm, "vgg11_bn")


def test_unknown_architecture_fails_clearly():
    with pytest.raises(KeyError, match="registered"):
        build_model("resnet50", 10)

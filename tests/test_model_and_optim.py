import pytest
import torch
import torch.nn as nn

from continuation.config import ModelConfig, OptimConfig
from continuation.models import build_model, count_parameters
from continuation.models.resnet_gn import ResNetCifarGN
from continuation.optim import build_optimizer, lr_at, reset_momentum, set_lr


def test_output_shape_matches_num_classes():
    for num_classes in (10, 100):
        m = build_model(ModelConfig(), num_classes=num_classes, seed=0)
        out = m(torch.zeros(4, 3, 32, 32))
        assert out.shape == (4, num_classes)


def test_depth_and_parameter_count_are_resnet20_sized():
    m = build_model(ModelConfig(), 10, seed=0)
    convs = [mod for mod in m.modules() if isinstance(mod, nn.Conv2d)]
    linears = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    assert len(convs) == 19 and len(linears) == 1        # 6n+2 = 20 weight layers
    n = count_parameters(m)["total"]
    assert 260_000 < n < 285_000
    assert count_parameters(m)["trainable"] == n


def test_groupnorm_configuration_and_absence_of_batchnorm():
    m = build_model(ModelConfig(channels_per_group=8), 10, seed=0)
    gns = [mod for mod in m.modules() if isinstance(mod, nn.GroupNorm)]
    assert gns, "expected GroupNorm layers"
    assert not [mod for mod in m.modules() if isinstance(mod, nn.modules.batchnorm._BatchNorm)]
    assert not list(m.buffers()), "GroupNorm must leave no running-statistic buffers"
    for gn in gns:
        assert gn.num_channels // gn.num_groups == 8
        assert gn.eps == 1e-5 and gn.affine
    widths = sorted({gn.num_channels for gn in gns})
    assert widths == [16, 32, 64]


def test_channels_per_group_must_divide_the_widths():
    with pytest.raises(ValueError):
        build_model(ModelConfig(channels_per_group=5), 10, seed=0)


def test_initialization_is_paired_across_conditions_with_the_same_seed():
    a = build_model(ModelConfig(), 10, seed=3)
    b = build_model(ModelConfig(), 10, seed=3)
    c = build_model(ModelConfig(), 10, seed=4)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    assert any(not torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters()))


def test_shortcut_option_b_adds_projection_parameters():
    a = count_parameters(build_model(ModelConfig(shortcut="A"), 10, seed=0))["total"]
    b = count_parameters(build_model(ModelConfig(shortcut="B"), 10, seed=0))["total"]
    assert b > a
    with pytest.raises(ValueError):
        ResNetCifarGN(shortcut="C")


def test_model_is_unmodified_internally_no_feature_smoothing():
    """The input-only baseline must not contain a feature-map filter."""
    m = build_model(ModelConfig(), 10, seed=0)
    names = " ".join(type(mod).__name__.lower() for mod in m.modules())
    assert "smooth" not in names and "blur" not in names and "gaussian" not in names


def test_eval_mode_output_does_not_depend_on_batch_composition():
    """A consequence of using GroupNorm rather than BatchNorm."""
    m = build_model(ModelConfig(), 10, seed=0).eval()
    g = torch.Generator().manual_seed(0)
    x = torch.rand(8, 3, 32, 32, generator=g)
    with torch.no_grad():
        full = m(x)
        singles = torch.cat([m(x[i: i + 1]) for i in range(8)])
    assert torch.allclose(full, singles, atol=1e-5)


def test_lr_schedule_is_indexed_by_global_step():
    cfg = OptimConfig(lr=0.1, warmup_steps=100, total_steps=1000, lr_schedule="cosine",
                      min_lr=0.0)
    assert lr_at(0, cfg) == pytest.approx(0.001)
    assert lr_at(99, cfg) == pytest.approx(0.1)
    assert lr_at(100, cfg) == pytest.approx(0.1)
    assert lr_at(999, cfg) < 1e-3
    seq = [lr_at(i, cfg) for i in range(100, 1000, 50)]
    assert all(a >= b for a, b in zip(seq, seq[1:]))    # monotone after warmup


def test_multistep_and_constant_schedules():
    cfg = OptimConfig(lr=0.1, warmup_steps=0, total_steps=1000, lr_schedule="multistep",
                      milestones=[500, 750], gamma=0.1)
    assert lr_at(0, cfg) == pytest.approx(0.1)
    assert lr_at(500, cfg) == pytest.approx(0.01)
    assert lr_at(750, cfg) == pytest.approx(0.001)
    cfg2 = OptimConfig(lr=0.05, warmup_steps=0, lr_schedule="constant")
    assert lr_at(0, cfg2) == lr_at(9999, cfg2) == pytest.approx(0.05)


def test_momentum_reset_is_explicit_and_separate_from_carrying():
    m = build_model(ModelConfig(), 10, seed=0)
    cfg = OptimConfig()
    opt = build_optimizer(m, cfg)
    set_lr(opt, 0.1)
    m(torch.zeros(2, 3, 32, 32)).sum().backward()
    opt.step()
    assert any("momentum_buffer" in s for s in opt.state.values())
    reset_momentum(opt)
    assert all("momentum_buffer" not in s for s in opt.state.values())

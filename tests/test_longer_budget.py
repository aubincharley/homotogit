"""The longer-budget arms: the stretch must be proportional, or it measures something else.

If the schedule boundaries did not scale with the budget, the intervention would
occupy a smaller share of a longer run and the comparison against the recorded
tables would be meaningless. These tests pin the arithmetic.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from continuation_core.controller import InterventionController         # noqa: E402
from continuation_core.models import build_model, site_map              # noqa: E402
from continuation_core.schedules import EpochSchedule                   # noqa: E402

import cifar10_configs                                                  # noqa: E402
import longer_budget_configs as L                                       # noqa: E402
import stl10_configs                                                    # noqa: E402


def test_stretch_scales_boundaries_and_keeps_values():
    s = EpochSchedule((0, 3, 6, 21), (1.0, 0.85, 0.7, 0.0))
    out = L.stretch(s, 3)
    assert out.starts == (0, 9, 18, 63)
    assert out.values == s.values


@pytest.mark.parametrize("dataset,k,base", [("cifar10", 3, 30), ("stl10", 2, 60)])
def test_budget_and_boundaries_scale_together(dataset, k, base):
    mod, kk, _, bb, _ = L.ARMS[dataset]
    assert (kk, bb) == (k, base)
    sgd = mod.build("resolution_max_b1_gaussian_conv", 0, data_root="d",
                    assets_dir="a", out_dir="r")
    lng = L.build(dataset, "resolution_max_b1_gaussian_conv", 0, data_root="d",
                  assets_dir="a", out_dir="r")
    assert lng.budget.epochs == sgd.budget.epochs * k == base * k
    a, b = sgd.method_spec(), lng.method_spec()
    for part in ("gaussian", "resolution"):
        sa, sb = getattr(a, part).schedule, getattr(b, part).schedule
        assert sb.starts == tuple(x * k for x in sa.starts)
        assert sb.values == sa.values          # only the timing moves


@pytest.mark.parametrize("dataset", sorted(L.ARMS))
def test_the_intervention_keeps_its_share_of_the_run(dataset):
    """The point of a proportional stretch: same fraction, longer run."""
    mod, k, _, base, _ = L.ARMS[dataset]
    sgd = mod.build("gaussian_postrelu", 0, data_root="d", assets_dir="a", out_dir="r")
    lng = L.build(dataset, "gaussian_postrelu", 0, data_root="d", assets_dir="a", out_dir="r")
    off_a = sgd.method_spec().gaussian.schedule.starts[-1] / sgd.budget.epochs
    off_b = lng.method_spec().gaussian.schedule.starts[-1] / lng.budget.epochs
    assert off_a == pytest.approx(off_b)


@pytest.mark.parametrize("dataset,opt", [("cifar10", "sgd"), ("stl10", "adam")])
def test_the_optimizer_is_the_one_being_extended(dataset, opt):
    cfg = L.build(dataset, "plain", 0, data_root="d", assets_dir="a", out_dir="r")
    assert cfg.optimizer.name == opt
    if opt == "adam":
        assert cfg.optimizer.lr == 1e-3 and cfg.optimizer.weight_decay == 0.0


@pytest.mark.parametrize("dataset", sorted(L.ARMS))
def test_the_dataset_s_own_decisions_survive(dataset):
    cfg = L.build(dataset, "resolution_max_b1_gaussian_conv", 0, data_root="d",
                  assets_dir="a", out_dir="r")
    spec = cfg.method_spec()
    if dataset == "stl10":
        assert spec.gaussian.schedule.values[0] == 3.00          # sigma x3 kept
        assert spec.resolution.schedule.values == (48, 72, 96)
        assert spec.resolution.reference_resolution == 96
    else:
        assert spec.gaussian.schedule.values[0] == 1.00
        assert spec.resolution.schedule.values == (16, 24, 32)


@pytest.mark.parametrize("dataset", sorted(L.ARMS))
@pytest.mark.parametrize("method_id", L.METHODS)
def test_gaussian_still_reaches_zero_and_resolution_native(dataset, method_id):
    cfg = L.build(dataset, method_id, 0, data_root="d", assets_dir="a", out_dir="r")
    spec, last = cfg.method_spec(), cfg.budget.epochs - 1
    if spec.gaussian:
        assert spec.gaussian.schedule.at(last) == 0.0
        assert spec.gaussian.schedule.at(spec.gaussian.schedule.starts[-1] - 1) > 0.0
    if spec.resolution:
        assert spec.resolution.schedule.at(last) == spec.resolution.reference_resolution


@pytest.mark.parametrize("method_id", L.METHODS)
def test_target_state_is_bitwise_plain(method_id):
    torch.manual_seed(0)
    arch = "resnet20_bn_cifar"
    m = build_model(arch, 10)
    cfg = L.build("cifar10", method_id, 0, data_root="d", assets_dir="a", out_dir="r")
    c = InterventionController(cfg.method_spec(), site_map(arch), arch)
    c.attach(m)
    plain = build_model(arch, 10)
    plain.load_state_dict(m.state_dict())
    c.set_state(c.target_state())
    m.eval(), plain.eval()
    x = torch.rand(4, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(m(x), plain(x))


def test_cifar_records_that_it_cannot_reuse_the_reference_data_order():
    """The pinned set has 30 epochs of order; 90 needs a new one. Say so."""
    cfg = L.build("cifar10", "plain", 0, data_root="d", assets_dir="a", out_dir="r")
    joined = " ".join(cfg.notes)
    assert "30 epochs of data order" in joined and "--init-from" in joined

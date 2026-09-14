"""The CIFAR-10 5,000-image arms: the control STL-10 is missing.

What must hold for the experiment to mean anything is arithmetic about *updates*,
not accuracy: ``stl_budget`` has to match STL-10's budget exactly and
``reference_updates`` has to match the CIFAR-10 reference's, or neither arm
isolates anything.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from continuation_core.controller import InterventionController         # noqa: E402
from continuation_core.methods import PLATEAU_G, RPROG                  # noqa: E402
from continuation_core.models import build_model, site_map              # noqa: E402
from continuation_core.presets import CIFAR10_MEAN, CIFAR10_STD         # noqa: E402

import cifar10_subset_configs as C                                      # noqa: E402

ARCH = "resnet20_bn_cifar"
REF_UPDATES, REF_RECOVERY, REF_PER_EPOCH = 11730, 3519, 391
STL_UPDATES, STL_RECOVERY = 2400, 720


def test_subset_gives_stl10s_updates_per_epoch():
    assert C.SUBSET_SIZE == 5000
    assert -(-C.SUBSET_SIZE // 128) == C.UPDATES_PER_EPOCH == 40


def test_stl_budget_arm_matches_stl10_exactly():
    a = C.ARMS["stl_budget"]
    assert a["epochs"] * C.UPDATES_PER_EPOCH == STL_UPDATES
    assert C.recovery_updates("stl_budget") == STL_RECOVERY
    # STL-10's own boundaries, which were the reference's doubled
    assert a["g_starts"] == tuple(2 * e for e in PLATEAU_G.starts)
    assert a["r_starts"] == tuple(2 * e for e in RPROG.starts)


def test_reference_updates_arm_lands_on_the_reference_update_counts():
    a = C.ARMS["reference_updates"]
    total = a["epochs"] * C.UPDATES_PER_EPOCH
    assert abs(total - REF_UPDATES) <= 20                  # 11,720 against 11,730
    assert abs(C.recovery_updates("reference_updates") - REF_RECOVERY) <= 20
    # every transition within half a reference epoch of where it should be
    for sub, ref in zip(a["g_starts"], PLATEAU_G.starts):
        assert abs(sub * C.UPDATES_PER_EPOCH - ref * REF_PER_EPOCH) <= REF_PER_EPOCH // 2
    for sub, ref in zip(a["r_starts"], RPROG.starts):
        assert abs(sub * C.UPDATES_PER_EPOCH - ref * REF_PER_EPOCH) <= REF_PER_EPOCH // 2


def test_the_two_arms_differ_only_in_timing():
    """Same intervention, different schedule -- otherwise nothing is isolated."""
    ga, ra = C.schedules("stl_budget")
    gb, rb = C.schedules("reference_updates")
    assert ga.values == gb.values == PLATEAU_G.values
    assert ra.values == rb.values == RPROG.values
    assert ga.starts != gb.starts


@pytest.mark.parametrize("arm", sorted(C.ARMS))
@pytest.mark.parametrize("method_id", C.METHODS)
def test_config_keeps_the_reference_recipe_and_pins(method_id, arm, tmp_path):
    cfg = C.build(method_id, 0, arm, data_root="d", assets_dir="a", out_dir="runs")
    assert cfg.validation_status == "unvalidated"
    assert cfg.budget.epochs == C.ARMS[arm]["epochs"]
    assert cfg.optimizer.lr == 0.005 and cfg.optimizer.weight_decay == 5e-4
    # the pipeline fits on all 50,000 before subsetting, so the reference
    # statistics still apply and are checked
    assert cfg.data.expected_mean == CIFAR10_MEAN
    assert cfg.data.expected_std == CIFAR10_STD
    spec = cfg.method_spec()
    if spec.gaussian:
        assert spec.gaussian.sigma_max == 1.0        # 32x32: no rescaling
        assert spec.gaussian.schedule.values == PLATEAU_G.values
    if spec.resolution:
        assert spec.resolution.reference_resolution == 32
        assert spec.resolution.schedule.values == (16, 24, 32)
    assert arm in cfg.run.name
    p = tmp_path / "c.json"
    cfg.save(p)
    assert type(cfg).load(p).to_dict() == cfg.to_dict()


@pytest.mark.parametrize("arm", sorted(C.ARMS))
def test_gaussian_reaches_zero_and_resolution_reaches_native(arm):
    epochs = C.ARMS[arm]["epochs"]
    g, r = C.schedules(arm)
    assert g.at(epochs - 1) == 0.0
    assert r.at(epochs - 1) == 32
    assert g.at(C.ARMS[arm]["g_starts"][-1] - 1) > 0.0     # off exactly at the boundary


@pytest.mark.parametrize("method_id", C.METHODS)
@pytest.mark.parametrize("arm", sorted(C.ARMS))
def test_target_state_is_bitwise_plain(method_id, arm):
    torch.manual_seed(0)
    model = build_model(ARCH, 10)
    cfg = C.build(method_id, 0, arm, data_root="d", assets_dir="a", out_dir="runs")
    ctrl = InterventionController(cfg.method_spec(), site_map(ARCH), ARCH)
    ctrl.attach(model)
    plain = build_model(ARCH, 10)
    plain.load_state_dict(model.state_dict())
    ctrl.set_state(ctrl.target_state())
    model.eval(), plain.eval()
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(model(x), plain(x))

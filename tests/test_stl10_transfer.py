"""The STL-10 transfer: schedules, 96x96 geometry, and the operators at radius 12.

Nothing in the existing suite runs at 96x96, and the two code paths this transfer
newly depends on -- a 25-tap kernel, and reflection padding where the pad equals
the map width -- are exercised on CIFAR-10 only at radius 4 on 4x4 maps.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from conftest import synthetic_dataset                                  # noqa: E402
from continuation_core.assets import make_assets, sha_state             # noqa: E402
from continuation_core.controller import InterventionController         # noqa: E402
from continuation_core.models import build_model, site_map              # noqa: E402
from continuation_core.operators import (FixedSupportGaussian,          # noqa: E402
                                         adaptive_windows,
                                         reflected_indices)

import stl10_configs as S                                               # noqa: E402

ARCH = "resnet20_bn_cifar"
CIFAR_REFERENCE = (1.00, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30, 0.0)


def _controller(method_id):
    cfg = S.build(method_id, 0, data_root="d", assets_dir="a", out_dir="runs")
    return InterventionController(cfg.method_spec(), site_map(ARCH), ARCH), cfg


# -- schedules ---------------------------------------------------------------

def test_schedules_are_the_reference_stretched_and_scaled():
    assert S.STL_G.starts == (0, 6, 12, 18, 24, 30, 36, 42)
    assert S.STL_G.values == tuple(round(v * 3.0, 10) for v in CIFAR_REFERENCE)
    assert S.STL_R.starts == (0, 12, 24) and S.STL_R.values == (48, 72, 96)
    # every level holds for twice as many epochs, and G is off for the same
    # fraction of the run as on CIFAR-10 (9/30 = 18/60)
    assert S.STL_G.table(S.EPOCHS) == [3.00] * 6 + [2.55] * 6 + [2.10] * 6 + \
        [1.80] * 6 + [1.50] * 6 + [1.20] * 6 + [0.90] * 6 + [0.0] * 18
    assert S.STL_R.table(S.EPOCHS) == [48] * 12 + [72] * 12 + [96] * 36
    assert S.STL_G.at(41) == 0.90 and S.STL_G.at(42) == 0.0


def test_ratios_of_the_resolution_schedule_are_the_reference_ratios():
    for stl, cifar in zip(S.STL_R.values, (16, 24, 32)):
        assert stl / S.NATIVE_RESOLUTION == cifar / 32


# -- per-site sigma ----------------------------------------------------------

def test_gaussian_postrelu_sigma_is_the_level_itself():
    ctrl, _ = _controller("gaussian_postrelu")
    assert ctrl.n_sites == 10
    for epoch, want in ((0, 3.00), (6, 2.55), (36, 0.90), (42, 0.0)):
        ctrl.set_epoch(epoch)
        assert ctrl.scale() == 1.0
        assert set(ctrl.per_site_sigma()) == {want}


def test_combined_sigma_is_three_times_the_cifar_table_and_non_monotone():
    ctrl, _ = _controller("resolution_max_b1_gaussian_conv")
    assert ctrl.n_sites == 19
    cifar = {0: 0.500, 6: 0.425, 12: 0.525, 18: 0.450,
             24: 0.500, 30: 0.400, 36: 0.300, 42: 0.0}
    for epoch, want in cifar.items():
        ctrl.set_epoch(epoch)
        got = set(ctrl.per_site_sigma())
        assert len(got) == 1
        assert got.pop() == pytest.approx(want * 3.0)
    # the r/96 factor makes the sequence rise again at epoch 12, as on CIFAR-10
    def sigma_at(epoch):
        ctrl.set_epoch(epoch)
        return ctrl.per_site_sigma()[0]
    assert sigma_at(12) > sigma_at(6)


def test_no_sigma_ever_exceeds_the_fixed_support():
    """sigma == sigma_max exactly at epoch 0 must not trip the guard."""
    for method_id in ("gaussian_postrelu", "resolution_max_b1_gaussian_conv"):
        ctrl, cfg = _controller(method_id)
        spec = cfg.method_spec().gaussian
        assert spec.sigma_max == 3.0 and spec.truncate == 4.0
        assert ctrl.gauss.radius == 12 and ctrl.gauss.kernel_size == 25
        x = torch.rand(2, 4, 16, 16)
        for e in range(S.EPOCHS):
            ctrl.set_epoch(e)
            s = ctrl.site_sigma(0)
            assert s <= spec.sigma_max
            ctrl.gauss(x, s)          # raises if the guard is tripped


# -- operators at 96x96 ------------------------------------------------------

def test_gaussian_on_the_smallest_map_uses_the_explicit_gather():
    """r=48 puts blocks.6-8 conv outputs at 12x12, where pad == n."""
    idx = reflected_indices(12, 12)
    assert idx.numel() == 36
    assert int(idx.min()) == 0 and int(idx.max()) == 11
    g = FixedSupportGaussian(sigma_max=3.0, truncate=4.0)
    x = torch.rand(2, 64, 12, 12)
    y = g(x, 1.5)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert not torch.equal(y, x)
    assert g(x, 0.0) is x                      # exact bypass


def test_adaptive_windows_keep_the_reference_structure():
    assert adaptive_windows(96, 48) == [(i, i + 2) for i in range(0, 96, 2)]

    def overlaps(n, r):
        w = adaptive_windows(n, r)
        return sum(1 for (_, b), (c, _) in zip(w, w[1:]) if c < b)
    assert overlaps(32, 16) == 0 and overlaps(96, 48) == 0
    assert overlaps(32, 24) == 16 and overlaps(96, 72) == 48   # same 4/3 pattern


@pytest.mark.parametrize("method_id", S.METHODS)
def test_target_state_is_bitwise_plain_at_96(method_id):
    torch.manual_seed(0)
    model = build_model(ARCH, 10)
    ctrl, _ = _controller(method_id)
    ctrl.attach(model)
    plain = build_model(ARCH, 10)
    plain.load_state_dict(model.state_dict())
    x = torch.rand(2, 3, 96, 96)
    ctrl.set_state(ctrl.target_state())
    for m in (model, plain):
        m.eval()
    with torch.no_grad():
        assert torch.equal(model(x), plain(x))


@pytest.mark.parametrize("resolution,expected", [
    (48, {1: 96, 2: 48, 3: 24, 6: 12, 8: 12}),
    (72, {1: 96, 2: 72, 3: 36, 6: 18, 8: 18}),
    (96, {1: 96, 2: 96, 3: 48, 6: 24, 8: 24}),
])
def test_reduction_grids_at_96(resolution, expected):
    model = build_model(ARCH, 10)
    ctrl, _ = _controller("resolution_max_b1")
    ctrl.attach(model)
    seen = {}
    for i in expected:
        model.blocks[i].register_forward_hook(
            lambda _m, _i, o, i=i: seen.__setitem__(i, o.shape[-1]))
    ctrl.set_epoch(S.STL_R.table(S.EPOCHS).index(resolution))
    assert model(torch.rand(2, 3, 96, 96)).shape == (2, 10)
    assert seen == expected


# -- configs and assets ------------------------------------------------------

@pytest.mark.parametrize("method_id", S.METHODS)
def test_configs_are_unvalidated_and_carry_every_decision(method_id, tmp_path):
    cfg = S.build(method_id, 0, data_root="d", assets_dir="a", out_dir="runs")
    assert cfg.validation_status == "unvalidated"
    assert cfg.method_spec().id.endswith("__transfer")
    assert cfg.budget.epochs == 60 and cfg.evaluation.every_epochs == 3
    assert cfg.optimizer.lr == 0.005 and cfg.optimizer.weight_decay == 5e-4
    assert cfg.checkpoint.every_epoch is False and cfg.checkpoint.keep_rolling
    r = cfg.method_spec().resolution
    assert r is None or r.reference_resolution == 96
    # plain evaluates one path: its current state IS the target state
    assert cfg.evaluation.paths == (("current",) if method_id == "plain"
                                    else ("current", "target"))
    path = tmp_path / "c.json"
    cfg.save(path)
    assert type(cfg).load(path).to_dict() == cfg.to_dict()


def test_make_assets_can_reuse_pinned_initial_weights(tmp_path):
    ds = synthetic_dataset(n_train=40, n_test=10, side=96)
    builder = lambda: build_model(ARCH, ds.num_classes)                # noqa: E731
    a = make_assets(tmp_path / "a", builder, n_train=len(ds.train), epochs=3, seeds=(0, 1))
    b = make_assets(tmp_path / "b", builder, n_train=len(ds.train), epochs=3, seeds=(0, 1),
                    init_from=tmp_path / "a")
    assert b["states"] == a["states"]
    assert b["arrays"] == a["arrays"]
    assert b["provenance"]["init_from"] == str(tmp_path / "a")
    assert b["paired_with_reference"] is False
    reloaded = torch.load(tmp_path / "b" / "init_seed0.pt", map_location="cpu",
                          weights_only=True)
    assert sha_state(reloaded) == a["states"]["init_seed0"]


def test_make_assets_rejects_initial_weights_of_another_shape(tmp_path):
    ds = synthetic_dataset(n_train=40, n_test=10, side=96)
    make_assets(tmp_path / "a", lambda: build_model(ARCH, 10), n_train=len(ds.train),
                epochs=2, seeds=(0,))
    with pytest.raises(RuntimeError):
        make_assets(tmp_path / "b", lambda: build_model(ARCH, 100), n_train=len(ds.train),
                    epochs=2, seeds=(0,), init_from=tmp_path / "a")

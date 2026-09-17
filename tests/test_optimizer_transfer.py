"""SVHN and STL-10 under Adam / AdamW: only the optimizer may differ.

The study is only meaningful if swapping the optimizer changes the optimizer and
nothing else. These tests pin that against the recorded SGD studies, and pin the
two facts about Adam that are easy to get wrong: it is not AdamW, and its
weight_decay does not mean what AdamW's does.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from continuation_core.config import OptimizerConfig                  # noqa: E402
from continuation_core.optim import build_optimizer, lr_at            # noqa: E402
from continuation_core.presets import CIFAR10_MEAN                    # noqa: E402

import optimizer_configs as O                                         # noqa: E402
import cifar10_configs                                                # noqa: E402
import stl10_configs                                                  # noqa: E402
import svhn_configs                                                   # noqa: E402


# -- the optimizer layer -------------------------------------------------------

def test_adam_and_adamw_are_distinct_classes():
    p = [torch.nn.Parameter(torch.zeros(2))]
    a = build_optimizer(p, OptimizerConfig(name="adam", lr=1e-3, weight_decay=0.0))
    w = build_optimizer(p, OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.01))
    assert isinstance(a, torch.optim.Adam) and not isinstance(a, torch.optim.AdamW)
    assert isinstance(w, torch.optim.AdamW)


def test_adam_still_refuses_borrowed_defaults():
    p = [torch.nn.Parameter(torch.zeros(2))]
    with pytest.raises(ValueError, match="explicit lr"):
        build_optimizer(p, OptimizerConfig(name="adam"))
    with pytest.raises(ValueError, match="explicit lr"):
        build_optimizer(p, OptimizerConfig(name="adam", lr=1e-3))


def test_unknown_optimizer_names_the_three_available():
    p = [torch.nn.Parameter(torch.zeros(2))]
    with pytest.raises(KeyError, match="sgd, adam, adamw"):
        build_optimizer(p, OptimizerConfig(name="lion", lr=1e-3, weight_decay=0.0))


def test_adam_decay_is_coupled_and_adamw_decay_is_not():
    """The same number means different things, which is why they are not matched."""
    def step(name, wd):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.ones(4) * 3.0)
        opt = build_optimizer([p], OptimizerConfig(name=name, lr=1e-2, weight_decay=wd))
        p.grad = torch.zeros(4)          # no data gradient: decay is the only force
        opt.step()
        return p.detach().clone()
    # with a zero gradient, AdamW still shrinks the weight; Adam's L2 enters the
    # gradient and is then normalised by the adaptive denominator
    assert not torch.allclose(step("adamw", 0.01), step("adamw", 0.0))
    assert not torch.allclose(step("adamw", 0.01), step("adam", 0.01))


# -- the lr schedule is shared machinery and must be untouched -----------------

@pytest.mark.parametrize("opt", sorted(O.OPTIMIZERS))
def test_warmup_and_cosine_are_the_recipe_s(opt):
    cfg = OptimizerConfig(**O.OPTIMIZERS[opt])
    assert cfg.schedule == "warmup_cosine" and cfg.warmup_updates == 60
    assert cfg.min_lr == 0.0
    assert lr_at(0, cfg, 11730) == pytest.approx(cfg.lr / 60)
    assert lr_at(59, cfg, 11730) == pytest.approx(cfg.lr)
    assert lr_at(11729, cfg, 11730) < cfg.lr * 1e-3          # cosine has decayed


def test_the_chosen_settings_are_the_textbook_ones():
    assert O.OPTIMIZERS["adam"]["lr"] == 1e-3
    assert O.OPTIMIZERS["adam"]["weight_decay"] == 0.0
    assert O.OPTIMIZERS["adamw"]["lr"] == 1e-3
    assert O.OPTIMIZERS["adamw"]["weight_decay"] == 0.01


# -- only the optimizer differs from the recorded SGD study --------------------

@pytest.mark.parametrize("dataset,module", [("cifar10", cifar10_configs),
                                           ("svhn", svhn_configs),
                                           ("stl10", stl10_configs)])
@pytest.mark.parametrize("optimizer", sorted(O.OPTIMIZERS))
@pytest.mark.parametrize("method_id", O.METHODS)
def test_everything_except_the_optimizer_matches_the_sgd_study(
        dataset, module, optimizer, method_id):
    sgd = module.build(method_id, 0, data_root="d", assets_dir="a", out_dir="runs")
    new = O.build(dataset, optimizer, method_id, 0,
                  data_root="d", assets_dir="a", out_dir="runs")
    a, b = sgd.to_dict(), new.to_dict()
    # the run name and the notes are expected to differ; nothing else may
    for section in ("data", "model", "method", "budget", "evaluation", "checkpoint",
                    "assets"):
        assert a[section] == b[section], section
    assert a["run"]["seed"] == b["run"]["seed"]
    assert sgd.optimizer.name == "sgd"
    assert new.optimizer.name == optimizer
    assert new.optimizer.lr == O.OPTIMIZERS[optimizer]["lr"]


@pytest.mark.parametrize("dataset", sorted(O.STUDIES))
def test_the_dataset_s_own_decisions_survive(dataset):
    cfg = O.build(dataset, "adamw", "resolution_max_b1_gaussian_conv", 0,
                  data_root="d", assets_dir="a", out_dir="runs")
    spec = cfg.method_spec()
    if dataset == "stl10":
        assert cfg.budget.epochs == 60                       # the x2 stretch
        assert spec.gaussian.schedule.values[0] == 3.00      # sigma x3
        assert spec.resolution.schedule.values == (48, 72, 96)
        assert spec.resolution.reference_resolution == 96
    else:
        assert cfg.budget.epochs == 30                       # the reference budget
        assert spec.gaussian.schedule.values[0] == 1.00      # unscaled
        assert spec.resolution.schedule.values == (16, 24, 32)
    if dataset == "cifar10":
        # the assets dir is the caller's, so assert the module's default and the
        # statistics it pins rather than what this test happened to pass in
        assert cifar10_configs.REFERENCE_ASSETS == "assets/cifar10_resnet20bn"
        assert cfg.data.expected_mean == list(CIFAR10_MEAN)


@pytest.mark.parametrize("dataset", sorted(O.STUDIES))
def test_run_names_distinguish_the_optimizer(dataset):
    names = {o: O.build(dataset, o, "plain", 0, data_root="d", assets_dir="a",
                        out_dir="runs").run.name for o in O.OPTIMIZERS}
    assert len(set(names.values())) == len(names)
    for o, n in names.items():
        assert o in n and dataset in n


def test_the_headroom_caveat_is_recorded_in_every_config():
    """The lr was chosen without calibration; the config has to say so."""
    for dataset in O.STUDIES:
        for optimizer in O.OPTIMIZERS:
            cfg = O.build(dataset, optimizer, "plain", 0, data_root="d",
                          assets_dir="a", out_dir="runs")
            joined = " ".join(cfg.notes)
            assert "NOT matched" in joined and "headroom" in joined


def test_cifar10_under_sgd_reproduces_the_reference_recipe():
    """The study module must be the frozen recipe, or the optimizer arm is not
    measuring the optimizer."""
    from continuation_core.presets import reference
    for method_id in cifar10_configs.METHODS:
        ref = reference(method_id, 0)
        mine = cifar10_configs.build(method_id, 0, data_root="data",
                                     assets_dir="assets/cifar10_resnet20bn",
                                     out_dir="runs")
        for f in ("lr", "weight_decay", "momentum", "nesterov", "schedule",
                  "warmup_updates", "min_lr", "name"):
            assert getattr(ref.optimizer, f) == getattr(mine.optimizer, f), f
        assert ref.budget.epochs == mine.budget.epochs == 30
        assert ref.data.expected_mean == mine.data.expected_mean
        assert ref.assets.dir == mine.assets.dir
        rs, ms = ref.method_spec(), mine.method_spec()
        assert (rs.gaussian is None) == (ms.gaussian is None)
        if rs.gaussian:
            assert rs.gaussian.schedule.values == ms.gaussian.schedule.values
            assert rs.gaussian.placement == ms.gaussian.placement
        if rs.resolution:
            assert rs.resolution.schedule.values == ms.resolution.schedule.values
            assert rs.resolution.point == ms.resolution.point

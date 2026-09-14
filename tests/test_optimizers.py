"""The optimizer axis: construction, naming, and the one contrast that matters.

These cover what the optimizer benchmark relies on and nothing else.  The
methods, schedules, sites and operators are untouched by this axis and keep
their own tests.
"""
import pytest
import torch

from continuation_core.config import ExperimentConfig, OptimizerConfig
from continuation_core.optim import OPTIMIZERS, build_optimizer, lr_at, optimizer_tag
from continuation_core.presets import reference, reference_optimizer

EXPECTED = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW, "radam": torch.optim.RAdam}


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_every_optimizer_builds_and_needs_explicit_settings(name):
    p = [torch.nn.Parameter(torch.zeros(2))]
    with pytest.raises(ValueError, match="explicit lr"):
        build_optimizer(p, OptimizerConfig(name=name))
    with pytest.raises(ValueError, match="explicit lr"):
        build_optimizer(p, OptimizerConfig(name=name, lr=1e-3))
    opt = build_optimizer(p, OptimizerConfig(name=name, lr=1e-3, weight_decay=5e-4))
    assert isinstance(opt, EXPECTED[name])


def test_unknown_optimizer_names_the_available_ones():
    p = [torch.nn.Parameter(torch.zeros(2))]
    with pytest.raises(KeyError, match="adamw"):
        build_optimizer(p, OptimizerConfig(name="lion", lr=1e-3, weight_decay=0.0))


def _step(name, *, weight_decay, steps=5):
    """Run identical updates on identical data and return the final weights."""
    torch.manual_seed(0)
    w0 = torch.randn(4, 3)
    p = torch.nn.Parameter(w0.clone())
    opt = build_optimizer([p], OptimizerConfig(name=name, lr=1e-2,
                                               weight_decay=weight_decay))
    g = torch.randn(4, 3)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        p.grad = g.clone()
        opt.step()
    return p.detach()


def test_adam_and_adamw_differ_only_through_the_decay_coupling():
    """The pair is a single-factor contrast, so it must be exercised.

    With weight decay off the two are the same algorithm and must agree
    bitwise; with decay on they must not, or the coupled/decoupled axis this
    benchmark measures would be silently absent.
    """
    assert torch.equal(_step("adam", weight_decay=0.0),
                       _step("adamw", weight_decay=0.0))
    assert not torch.equal(_step("adam", weight_decay=5e-4),
                           _step("adamw", weight_decay=5e-4))


def test_adam_coupling_is_pinned_not_inherited_from_the_torch_default():
    """``decoupled_weight_decay`` is passed explicitly, so a torch default flip
    cannot silently turn Adam into AdamW between here and Kaggle."""
    p = [torch.nn.Parameter(torch.zeros(2))]
    for name in ("adam", "radam"):
        opt = build_optimizer(p, OptimizerConfig(name=name, lr=1e-3, weight_decay=5e-4))
        assert opt.param_groups[0]["decoupled_weight_decay"] is False


def test_lr_schedule_is_shared_and_unchanged_by_the_optimizer():
    total = 11730
    sgd = reference_optimizer()
    adam = OptimizerConfig(name="adam", lr=0.005, weight_decay=5e-4,
                           schedule="warmup_cosine", warmup_updates=60, min_lr=0.0)
    for u in (0, 59, 60, 6000, total):
        assert lr_at(u, sgd, total) == lr_at(u, adam, total)


def test_tag_separates_two_learning_rates_of_one_optimizer():
    """The sweep runs one optimizer, method and seed at several learning rates;
    without the lr in the tag those four runs share a directory."""
    tags = {optimizer_tag(OptimizerConfig(name="adam", lr=lr, weight_decay=5e-4))
            for lr in (3e-4, 1e-3, 3e-3, 1e-2)}
    assert len(tags) == 4
    assert optimizer_tag(reference_optimizer()) == "sgd_lr0.005"
    assert optimizer_tag(OptimizerConfig(name="adam", lr=1e-3,
                                         weight_decay=5e-4)) == "adam_lr0.001"
    with pytest.raises(ValueError, match="explicit lr"):
        optimizer_tag(OptimizerConfig(name="adam"))


def test_run_names_of_the_whole_grid_are_distinct():
    methods = ("plain", "resolution_max_b1", "gaussian_postrelu",
               "resolution_max_b1_gaussian_conv")
    opts = [reference_optimizer()] + [
        OptimizerConfig(name=n, lr=lr, weight_decay=5e-4)
        for n in ("adam", "adamw", "radam") for lr in (3e-4, 1e-3, 3e-3, 1e-2)]
    names = {reference(m, s, optimizer=o).run.name
             for m in methods for s in (0, 1, 2) for o in opts}
    assert len(names) == len(methods) * 3 * len(opts)


def test_only_the_untouched_recipe_claims_to_be_the_reference():
    assert reference("plain").validation_status == "reference"
    assert reference("plain", optimizer=reference_optimizer()).validation_status \
        == "reference"
    variant = reference("plain", optimizer=OptimizerConfig(name="adam", lr=1e-3,
                                                           weight_decay=5e-4))
    assert variant.validation_status == "optimizer-variant"
    assert any("optimizer replaced" in n for n in variant.notes)


def test_a_variant_changes_the_optimizer_and_nothing_else():
    base = reference("resolution_max_b1_gaussian_conv", 1).to_dict()
    var = reference("resolution_max_b1_gaussian_conv", 1,
                    optimizer=OptimizerConfig(name="radam", lr=1e-3,
                                              weight_decay=5e-4)).to_dict()
    differing = {k for k in base if base[k] != var[k]}
    assert differing == {"optimizer", "run", "validation_status", "notes"}
    assert {k: v for k, v in base["run"].items() if k != "name"} \
        == {k: v for k, v in var["run"].items() if k != "name"}


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_config_roundtrip_with_each_optimizer(tmp_path, name):
    cfg = reference("plain", optimizer=OptimizerConfig(name=name, lr=1e-3,
                                                       weight_decay=5e-4))
    cfg.save(tmp_path / "c.json")
    assert ExperimentConfig.load(tmp_path / "c.json").to_dict() == cfg.to_dict()

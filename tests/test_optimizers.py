"""The optimizer axis: construction, naming, and the one contrast that matters.

These cover what the optimizer benchmark relies on and nothing else.  The
methods, schedules, sites and operators are untouched by this axis and keep
their own tests.
"""
import sys

import pytest
import torch

from continuation_core import assets as assets_mod
from continuation_core.config import ExperimentConfig, OptimizerConfig
from continuation_core.models import build_model
from continuation_core.optim import OPTIMIZERS, build_optimizer, lr_at, optimizer_tag
from continuation_core.presets import reference, reference_optimizer
from continuation_core.train import Trainer

from conftest import ROOT, synthetic_dataset

sys.path.insert(0, str(ROOT / "scripts"))

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


def test_a_full_run_under_adam_records_the_optimizer(tmp_path):
    """One short end-to-end run: the aggregation reads summary.json alone, so the
    optimizer has to be in there and has to be the one that actually ran."""
    ds = synthetic_dataset()
    assets_mod.make_assets(tmp_path / "assets",
                           lambda: build_model("resnet20_bn_cifar", 10),
                           n_train=len(ds.train), epochs=2, seeds=(0,), probe_size=20)
    cfg = reference("resolution_max_b1", 0, assets_dir=str(tmp_path / "assets"),
                    device="cpu", out_dir=str(tmp_path / "runs"),
                    optimizer=OptimizerConfig(name="adam", lr=1e-3, weight_decay=5e-4,
                                              schedule="warmup_cosine",
                                              warmup_updates=60, min_lr=0.0))
    cfg.data.expected_mean = cfg.data.expected_std = None
    cfg.budget.epochs, cfg.budget.effective_batch, cfg.budget.microbatch = 2, 32, 16
    cfg.evaluation.batch_size = 20

    trainer = Trainer(cfg, dataset=ds, device="cpu")
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    summary = trainer.run()

    assert summary["optimizer"] == {
        "tag": "adam_lr0.001", "name": "adam", "lr": 1e-3, "weight_decay": 5e-4,
        "schedule": "warmup_cosine", "warmup_updates": 60, "min_lr": 0.0,
        "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay_coupling": "coupled_l2"}
    assert summary["validation_status"] == "optimizer-variant"
    assert (tmp_path / "runs" / "resolution_max_b1__adam_lr0.001__seed0"
            / "summary.json").is_file()


def test_the_campaign_manifest_covers_every_cell_exactly_once():
    """Kernels slice the manifest by index, so a slicing bug either loses a cell
    or, worse, runs one twice and silently overwrites its result."""
    import job_optimizer_benchmark as job

    job.CHOSEN_LR = {n: 1e-3 for n in job.NEW_OPTIMIZERS}
    sizes = {b: len(job.build_cells(b))
             for b in ("lr_sweep", "sgd_control", "grid")}
    assert sizes == {"lr_sweep": 12, "sgd_control": 3, "grid": 36}

    grid = job.build_cells("grid")
    for n_jobs in (1, 2, 3, 4):
        seen = sorted(c["index"] for j in range(n_jobs)
                      for c in job.slice_for(grid, j, n_jobs))
        assert seen == list(range(len(grid)))

    names = {job.config_for(c, "data", "assets", "runs").run.name for c in grid}
    assert len(names) == len(grid)


def test_every_arm_shares_the_reference_lr_schedule():
    """Only the optimizer may differ between arms.  If the warmup or the cosine
    moved with it, an optimizer contrast would also be a schedule contrast."""
    import job_optimizer_benchmark as job

    job.CHOSEN_LR = {n: 1e-3 for n in job.NEW_OPTIMIZERS}
    ref = reference_optimizer()
    for batch in ("lr_sweep", "sgd_control", "grid"):
        for cell in job.build_cells(batch):
            o = job.config_for(cell, "data", "assets", "runs").optimizer
            assert (o.schedule, o.warmup_updates, o.min_lr) == \
                (ref.schedule, ref.warmup_updates, ref.min_lr)
            assert o.weight_decay == ref.weight_decay


def test_the_grid_refuses_to_run_before_the_sweep_is_read_out():
    import job_optimizer_benchmark as job

    saved, job.CHOSEN_LR = job.CHOSEN_LR, {}
    try:
        with pytest.raises(SystemExit, match="lr_sweep"):
            job.build_cells("grid")
    finally:
        job.CHOSEN_LR = saved


def test_the_sgd_control_is_the_untouched_reference_recipe():
    """It is the drift control for the imported SGD numbers, so it must be the
    same configuration those numbers came from -- not a variant of it."""
    import job_optimizer_benchmark as job

    for cell in job.build_cells("sgd_control"):
        cfg = job.config_for(cell, "data", "assets", "runs")
        assert cfg.validation_status == "reference"
        assert cfg.optimizer == reference_optimizer()

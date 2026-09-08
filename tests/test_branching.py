"""Checks required before Experiment 1's branched runs (spec section 3).

These verify that branching at a mid-epoch update reproduces uninterrupted
training exactly, that stopping early does not compress the learning-rate
horizon, that sigma=0 really supplies original images, and that target-metric
evaluation is not coupled to the currently active training transform.
"""
import json

import pytest
import torch

from continuation.config import from_dict
from continuation.engine import Trainer
from continuation.metrics import checkpoint_kind, read_metrics
from continuation.optim import lr_at
from continuation.pipeline import ChannelNormalizer, InputPipeline
from continuation.transforms import GaussianSmoothing

from test_config_and_engine import synthetic_bundle


B = 40          # stand-in for the real 14,040-update horizon
BRANCH = 25     # mid-epoch: 256 images / batch 32 = 8 batches per epoch


def base_cfg(tmp_path, schedule, total=B):
    return from_dict({
        "run": {"name": "br", "seed": 0, "device": "cpu", "out_dir": str(tmp_path),
                "log_every": 0},
        "model": {"width": 8, "channels_per_group": 4},
        "optim": {"batch_size": 32, "total_steps": total, "warmup_steps": 4, "lr": 0.05},
        "transform": {"family": "gaussian", "params": {"sigma_max": 3.0, "truncate": 4.0}},
        "schedule": schedule,
        "evaluation": {"eval_every": 0, "train_probe_size": 32, "eval_batch_size": 32,
                       "transform_stats_size": 16, "dense_every": 0,
                       "eval_at_step_zero": False},
    })


def params(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_branch_point_is_mid_epoch():
    """Otherwise the resume test would not exercise the interesting case."""
    assert BRANCH % (256 // 32) != 0


def test_resume_reproduces_uninterrupted_training_exactly(tmp_path):
    bundle = synthetic_bundle()
    sched = {"kind": "constant", "params": {"value": 1.0}}

    uninterrupted = Trainer(base_cfg(tmp_path / "u", sched), bundle, tmp_path / "u")
    uninterrupted.fit()
    ref = params(uninterrupted.model)

    prefix = Trainer(base_cfg(tmp_path / "p", sched), bundle, tmp_path / "p")
    prefix.fit(max_steps=BRANCH)
    ckpt = tmp_path / "p" / ("checkpoint_step%06d.pt" % BRANCH)
    assert checkpoint_kind(ckpt) == "full_resumable"

    resumed = Trainer(base_cfg(tmp_path / "r", sched), bundle, tmp_path / "r",
                      resume_from=ckpt)
    assert resumed.start_step == BRANCH
    resumed.fit()
    got = params(resumed.model)

    for k in ref:
        assert torch.allclose(ref[k], got[k], atol=0, rtol=0), (
            "parameter %s differs after resume; branching is not exact" % k)


def test_prefix_does_not_compress_the_learning_rate_horizon(tmp_path):
    """A 25-update prefix must use the LR of the full B-update schedule."""
    cfg = base_cfg(tmp_path / "h", {"kind": "constant", "params": {"value": 1.0}})
    t = Trainer(cfg, synthetic_bundle(), tmp_path / "h")
    t.fit(max_steps=BRANCH)
    # LR at the last prefix update is the full-horizon value, not the value a
    # schedule with total_steps=BRANCH would have produced.
    full = lr_at(BRANCH - 1, cfg.optim)
    compressed_cfg = from_dict({"optim": {"total_steps": BRANCH, "warmup_steps": 4,
                                          "lr": 0.05}})
    compressed = lr_at(BRANCH - 1, compressed_cfg.optim)
    assert full > compressed
    assert t.cfg.optim.total_steps == B


def test_sigma_zero_supplies_original_images(tmp_path):
    bundle = synthetic_bundle()
    pipe = InputPipeline(GaussianSmoothing(sigma_max=3.0),
                         ChannelNormalizer(bundle.mean, bundle.std))
    raw = bundle.val.images[:16]
    direct = pipe.normalizer(InputPipeline.to_unit_float(raw))
    assert torch.equal(pipe(raw, 0.0), direct)


def test_transition_logs_pre_and_post_records_with_identical_target_metrics(tmp_path):
    """The section-3 invariance check, exercised through a real run."""
    sched = {"kind": "piecewise_constant",
             "params": {"values": [1.0, 0.5, 0.0], "steps_per_stage": [10, 10, 20]}}
    cfg = base_cfg(tmp_path / "t", sched)
    cfg.evaluation.eval_every = 5
    trainer = Trainer(cfg, synthetic_bundle(), tmp_path / "t")
    trainer.fit()

    recs = [r for r in read_metrics(tmp_path / "t" / "metrics.jsonl") if r["record"] == "eval"]
    pre = {r["step"]: r for r in recs if r.get("phase") == "pre_transition"}
    post = {r["step"]: r for r in recs if r.get("phase") == "post_transition"}
    assert set(pre) == set(post) == {10, 20}, "expected paired records at both transitions"

    for step in pre:
        a, b = pre[step], post[step]
        assert a["parameter"] != b["parameter"]          # active level changed
        for block in ("target_train_probe", "target_val"):
            assert a[block] == b[block], "target metrics moved with the active sigma"
        # the active-level metrics are the ones allowed to differ
        assert a["transformed_val"] != b["transformed_val"]


def test_target_invariance_guard_actually_fires():
    pre = {"step": 10,
           "target_train_probe": {"ce": 1.0, "accuracy": 0.5},
           "target_val": {"ce": 1.0, "accuracy": 0.5}}
    post = {"step": 10,
            "target_train_probe": {"ce": 1.0, "accuracy": 0.5},
            "target_val": {"ce": 1.02, "accuracy": 0.5}}
    with pytest.raises(AssertionError, match="coupled to the"):
        Trainer._assert_target_invariance(pre, post)


def test_transition_steps_and_eval_grid(tmp_path):
    sched = {"kind": "piecewise_constant",
             "params": {"values": [1.0, 0.5, 0.0], "steps_per_stage": [10, 10, 20]}}
    cfg = base_cfg(tmp_path / "g", sched)
    cfg.evaluation.eval_every = 10
    cfg.evaluation.dense_every = 2
    cfg.evaluation.dense_window = 6
    t = Trainer(cfg, synthetic_bundle(), tmp_path / "g")
    assert t.transition_steps() == [10, 20]
    grid = t.eval_step_grid(B)
    assert {10, 20, B} <= grid
    assert {12, 14, 16, 22, 24, 26} <= grid          # dense post-transition points


def test_momentum_policy_is_recorded_and_defaults_to_carry(tmp_path):
    cfg = base_cfg(tmp_path / "m", {"kind": "constant", "params": {"value": 0.0}})
    t = Trainer(cfg, synthetic_bundle(), tmp_path / "m")
    assert t.momentum_policy == "carry"
    assert t.describe()["momentum_at_stage_boundary"] == "carry"
    # the policy key sits next to the schedule config but is consumed by the
    # trainer, so it must not be forwarded to the schedule constructor
    cfg2 = base_cfg(tmp_path / "m2", {"kind": "constant",
                                      "params": {"value": 0.0,
                                                 "momentum_at_stage_boundary": "reset"}})
    t2 = Trainer(cfg2, synthetic_bundle(), tmp_path / "m2")
    assert t2.momentum_policy == "reset"
    cfg3 = base_cfg(tmp_path / "m3", {"kind": "constant",
                                      "params": {"value": 0.0,
                                                 "momentum_at_stage_boundary": "nonsense"}})
    with pytest.raises(ValueError, match="carry"):
        Trainer(cfg3, synthetic_bundle(), tmp_path / "m3")


def test_lineage_is_recorded_on_a_branch(tmp_path):
    bundle = synthetic_bundle()
    sched = {"kind": "constant", "params": {"value": 1.0}}
    prefix = Trainer(base_cfg(tmp_path / "l", sched), bundle, tmp_path / "l")
    prefix.fit(max_steps=BRANCH)
    ckpt = tmp_path / "l" / ("checkpoint_step%06d.pt" % BRANCH)
    child = Trainer(base_cfg(tmp_path / "c", sched), bundle, tmp_path / "c", resume_from=ckpt)
    assert child.lineage["parent_checkpoint"] == str(ckpt)
    assert child.lineage["branched_at_step"] == BRANCH
    desc = json.loads((tmp_path / "l" / "run_description.json").read_text())
    assert desc["start_step"] == 0

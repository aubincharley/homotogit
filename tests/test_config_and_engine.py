"""Config strictness and an end-to-end trainer smoke test on synthetic data.

No dataset download happens here: a tiny synthetic ``DatasetBundle`` exercises
the same code path the CIFAR runs use.
"""
import json

import numpy as np
import pytest
import torch

from continuation.config import ExperimentConfig, from_dict, load_config, parse_override, to_dict
from continuation.data import DatasetBundle, Split, channel_stats
from continuation.engine import Trainer, evaluate
from continuation.metrics import RunLogger, read_metrics
from continuation.pipeline import ChannelNormalizer, InputPipeline
from continuation.transforms import GaussianSmoothing


# ---------------------------------------------------------------- config ---

def test_unknown_keys_are_rejected_at_every_level():
    with pytest.raises(KeyError, match="unknown"):
        from_dict({"optim": {"learning_rate": 0.1}})
    with pytest.raises(KeyError, match="unknown"):
        from_dict({"optimiser": {}})


def test_defaults_and_roundtrip():
    cfg = from_dict({})
    assert cfg.optim.batch_size == 128 and cfg.optim.momentum == 0.9
    assert cfg.transform.family == "gaussian" and cfg.schedule.kind == "constant"
    assert from_dict(to_dict(cfg)) == cfg


def test_override_parsing_is_typed():
    assert parse_override("optim.lr=0.05") == {"optim": {"lr": 0.05}}
    assert parse_override("run.deterministic=false") == {"run": {"deterministic": False}}
    assert parse_override("sweep.levels=[0, 0.5, 1]") == {"sweep": {"levels": [0, 0.5, 1]}}
    with pytest.raises(ValueError):
        parse_override("nonsense")


def test_shipped_config_loads_and_matches_the_documented_grid():
    cfg = load_config("configs/exp0_gaussian.yaml")
    assert cfg.transform.family == "gaussian"
    assert cfg.sweep["levels"] == [0, 0.5, 1, 2, 3]
    assert len(cfg.sweep["seeds"]) == 3
    # the fixed support must cover the largest level in the grid
    assert cfg.transform.params["sigma_max"] >= max(cfg.sweep["levels"])


# ---------------------------------------------------------------- engine ---

def synthetic_bundle(n_train=256, n_val=128, n_classes=4, seed=0):
    g = torch.Generator().manual_seed(seed)

    def make(n, offset):
        labels = torch.arange(n) % n_classes
        base = torch.rand(n_classes, 3, 32, 32, generator=g)
        imgs = (base[labels] * 200 + torch.randint(0, 40, (n, 3, 32, 32), generator=g)).clamp(0, 255)
        return Split("s", imgs.to(torch.uint8), labels, np.arange(offset, offset + n))

    train = make(n_train, 0)
    val = make(n_val, n_train)
    test = make(n_val, n_train + n_val)
    mean, std = channel_stats(train.images)
    return DatasetBundle(train, val, test, n_classes, [str(i) for i in range(n_classes)], mean, std)


def tiny_config(tmp_path, level=1.0, steps=6):
    return from_dict({
        "run": {"name": "smoke", "seed": 0, "device": "cpu", "out_dir": str(tmp_path),
                "log_every": 2},
        "model": {"width": 8, "channels_per_group": 4},
        "optim": {"batch_size": 32, "total_steps": steps, "warmup_steps": 2, "lr": 0.05},
        "transform": {"family": "gaussian", "params": {"sigma_max": 3.0, "truncate": 4.0}},
        "schedule": {"kind": "constant", "params": {"value": level}},
        "evaluation": {"eval_every": 3, "train_probe_size": 64, "eval_batch_size": 64,
                       "transform_stats_size": 32, "save_final_checkpoint": False},
    })


def test_trainer_runs_and_logs_both_objectives(tmp_path):
    cfg = tiny_config(tmp_path, level=1.0)
    bundle = synthetic_bundle()
    logger = RunLogger(tmp_path / "run")
    summary = Trainer(cfg, bundle, tmp_path / "run", logger).fit()
    logger.close()

    assert summary["parameter"] == 1.0
    final = summary["final"]
    for block in ("transformed_train_probe", "transformed_val",
                  "target_train_probe", "target_val"):
        assert final[block]["ce"] > 0 and 0.0 <= final[block]["accuracy"] <= 1.0
    # the two objectives differ when sigma > 0
    assert final["transformed_val"]["ce"] != final["target_val"]["ce"]

    records = read_metrics(tmp_path / "run" / "metrics.jsonl")
    kinds = {r["record"] for r in records}
    assert {"train_minibatch", "eval", "final", "transform_stats"} <= kinds
    mb = [r for r in records if r["record"] == "train_minibatch"]
    assert all("minibatch_transformed_ce_mean" in r for r in mb)
    assert all("not an eval-mode metric" in r["note"] for r in mb)

    desc = json.loads((tmp_path / "run" / "run_description.json").read_text())
    assert desc["seeds"]["init_stream"] != desc["seeds"]["batch_stream"]
    assert desc["environment"]["torch"]
    assert desc["pipeline"]["order"][1].startswith("T_eta")


def test_at_the_target_endpoint_both_objectives_coincide(tmp_path):
    cfg = tiny_config(tmp_path, level=0.0)
    summary = Trainer(cfg, synthetic_bundle(), tmp_path / "run0").fit()
    final = summary["final"]
    assert final["is_target_endpoint"] is True
    assert final["transformed_val"] == final["target_val"]
    assert final["transformed_train_probe"] == final["target_train_probe"]
    stats = summary["transform_stats"]["start"]
    assert stats["reconstruction_mse"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert stats["retained_tv_ratio"]["mean"] == pytest.approx(1.0, abs=1e-12)


def test_paired_seeds_give_identical_batches_across_levels(tmp_path):
    """Different transformation levels must consume the same sample indices."""
    from continuation.data import BatchIndexStream

    a = BatchIndexStream(256, 32, seed=0)
    b = BatchIndexStream(256, 32, seed=0)
    t = GaussianSmoothing(sigma_max=3.0)
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        t(torch.rand(2, 3, 32, 32, generator=g), 2.0)     # deterministic, draws nothing
        assert np.array_equal(a.next_indices(), b.next_indices())


def test_evaluate_is_eval_mode_and_deterministic(tmp_path):
    from continuation.models import build_model

    bundle = synthetic_bundle(n_train=64, n_val=64)
    model = build_model(tiny_config(tmp_path).model, bundle.num_classes, seed=0)
    pipe = InputPipeline(GaussianSmoothing(sigma_max=3.0),
                         ChannelNormalizer(bundle.mean, bundle.std))
    model.train()
    r1 = evaluate(model, bundle.val.images, bundle.val.labels, pipe, 1.0, batch_size=16)
    r2 = evaluate(model, bundle.val.images, bundle.val.labels, pipe, 1.0, batch_size=32)
    assert model.training, "evaluate must restore the previous mode"
    assert r1.ce == pytest.approx(r2.ce, rel=1e-5)   # batch size must not matter
    assert r1.n == 64


def test_transform_cost_is_accounted_separately(tmp_path):
    cfg = tiny_config(tmp_path, level=2.0, steps=4)
    trainer = Trainer(cfg, synthetic_bundle(), tmp_path / "run2")
    summary = trainer.fit()
    timing = summary["timing"]
    assert timing["transform_seconds_train"] > 0
    assert timing["train_seconds"] >= timing["transform_seconds_train"]
    assert "not the total computational cost" in timing["note"]

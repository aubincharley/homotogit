import math
import pickle

import numpy as np
import pytest
import torch

from continuation_core.config import DataConfig, ExperimentConfig, OptimizerConfig
from continuation_core.data import build_pipeline, load_dataset
from continuation_core.methods import RPROG
from continuation_core.optim import build_optimizer, lr_at
from continuation_core.presets import reference, transfer
from continuation_core.schedules import EpochSchedule


def test_reference_preset_is_the_recipe(tmp_path):
    cfg = reference("resolution_max_b1", 2)
    o, b = cfg.optimizer, cfg.budget
    assert (o.name, o.lr, o.momentum, o.weight_decay, o.nesterov, o.warmup_updates) == \
        ("sgd", 0.005, 0.9, 5e-4, False, 60)
    assert (b.epochs, b.effective_batch, b.microbatch) == (30, 128, 32)
    assert cfg.validation_status == "reference" and cfg.model.arch == "resnet20_bn_cifar"
    cfg.save(tmp_path / "c.json")
    assert ExperimentConfig.load(tmp_path / "c.json").to_dict() == cfg.to_dict()
    with pytest.raises(KeyError):
        ExperimentConfig.from_dict({**cfg.to_dict(), "surprise": 1})


def test_lr_schedule_matches_the_benchmark_formula():
    cfg = reference("plain").optimizer
    total = 11730
    assert lr_at(0, cfg, total) == 0.005 / 60
    assert lr_at(59, cfg, total) == 0.005
    assert lr_at(60, cfg, total) == 0.005
    mid = 60 + (total - 60) // 2
    assert math.isclose(lr_at(mid, cfg, total), 0.0025, rel_tol=1e-3)
    assert lr_at(total, cfg, total) == pytest.approx(0.0, abs=1e-15)


def test_adamw_needs_its_own_settings():
    p = [torch.nn.Parameter(torch.zeros(2))]
    with pytest.raises(ValueError, match="explicit lr"):
        build_optimizer(p, OptimizerConfig(name="adamw"))
    opt = build_optimizer(p, OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.05))
    assert isinstance(opt, torch.optim.AdamW)
    with pytest.raises(KeyError):
        build_optimizer(p, OptimizerConfig(name="lion", lr=1e-3, weight_decay=0.0))


def test_transfer_requires_every_decision():
    opt = OptimizerConfig(name="adamw", lr=None, weight_decay=None)
    with pytest.raises(ValueError, match="resolution_schedule"):
        transfer("resolution_max_b1", dataset="stl10", data_root="d", arch="resnet20_bn_cifar",
                 optimizer=opt, epochs=30, resolution_schedule=None, reference_resolution=None,
                 gaussian_schedule=None, gaussian_units=None, insertion_mapping_note="x",
                 assets_dir="a")
    cfg = transfer("resolution_max_b1", dataset="stl10", data_root="d", arch="resnet20_bn_cifar",
                   optimizer=opt, epochs=30,
                   resolution_schedule=EpochSchedule((0, 6, 12), (48, 72, 96)),
                   reference_resolution=96, gaussian_schedule=None, gaussian_units=None,
                   insertion_mapping_note="blocks.2 input, as in the reference",
                   assets_dir="a")
    assert cfg.validation_status == "unvalidated"
    assert cfg.method_spec().resolution.schedule.values == (48, 72, 96)


def test_stl10_binary_is_read_row_major(tmp_path):
    base = tmp_path / "stl10_binary"
    base.mkdir()
    img = np.arange(3 * 96 * 96, dtype=np.uint32).reshape(3, 96, 96) % 251
    img = img.astype(np.uint8)
    col_major = np.stack([img, img[:, ::-1]]).transpose(0, 1, 3, 2)   # as stored on disk
    for split in ("train", "test"):
        col_major.tofile(base / ("%s_X.bin" % split))
        np.array([1, 10], dtype=np.uint8).tofile(base / ("%s_y.bin" % split))
    ds = load_dataset(DataConfig(name="stl10", root=str(tmp_path)))
    assert torch.equal(ds.train.images[0], torch.from_numpy(img))
    assert ds.train.labels.tolist() == [0, 9] and ds.native_resolution == 96


def test_cifar10_reader_and_pipeline(tmp_path):
    base = tmp_path / "cifar-10-batches-py"
    base.mkdir()
    rng = np.random.default_rng(0)
    for name in ["data_batch_%d" % i for i in range(1, 6)] + ["test_batch"]:
        with open(base / name, "wb") as fh:
            pickle.dump({"data": rng.integers(0, 256, (4, 3072), dtype=np.uint8),
                         "labels": list(range(4))}, fh)
    with open(base / "batches.meta", "wb") as fh:
        pickle.dump({"label_names": [str(i) for i in range(10)]}, fh)
    ds = load_dataset(DataConfig(name="cifar10", root=str(tmp_path)))
    assert ds.train.images.shape == (20, 3, 32, 32) and ds.test.images.shape == (4, 3, 32, 32)
    pipe = build_pipeline(ds, DataConfig())
    x = pipe(ds.train.images)
    assert torch.allclose(x.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-5)
    with pytest.raises(ValueError, match="pinned"):
        build_pipeline(ds, DataConfig(expected_mean=[0.1, 0.2, 0.3], expected_std=[1, 1, 1]))

"""Datasets as uint8 tensors, and the input pipeline.

Registered datasets
-------------------
``cifar10``  ``<root>/cifar-10-batches-py`` (the official python pickles);
             data_batch_1..5 concatenated in order, then test_batch.  This is
             the order torchvision uses, and the order the pinned subset
             indices refer to (checked bitwise in verification/parity.py).
``stl10``    ``<root>/stl10_binary`` (official binaries); labelled train 5,000
             and test 8,000 images at 96x96.  Stored column-major, so each
             image is transposed to row-major.  Labels 1..10 become 0..9.
             Unlabelled images are not loaded.  No reference result exists.

Pipeline (reference, identical to the benchmark)
------------------------------------------------
``uint8 -> float32 / 255 -> (x - mean) / std`` with per-channel statistics fitted
once on the **unfiltered training images** (float64, population std, cast to
float32).  No augmentation, no resize: the retained interventions act inside
the network.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import DataConfig


@dataclass
class Split:
    images: torch.Tensor          # uint8 [N, C, H, W]
    labels: torch.Tensor          # int64 [N]

    def to(self, device) -> "Split":
        return Split(self.images.to(device), self.labels.to(device))

    def __len__(self):
        return int(self.images.shape[0])


@dataclass
class Dataset:
    name: str
    train: Split
    test: Split
    num_classes: int
    native_resolution: int
    class_names: list

    def to(self, device) -> "Dataset":
        return Dataset(self.name, self.train.to(device), self.test.to(device),
                       self.num_classes, self.native_resolution, self.class_names)


def _cifar10(root: Path) -> Dataset:
    base = root / "cifar-10-batches-py"
    if not base.is_dir():
        raise FileNotFoundError("CIFAR-10 python batches not found under %s" % base)

    def load(name):
        with open(base / name, "rb") as fh:
            entry = pickle.load(fh, encoding="latin1")
        return (np.asarray(entry["data"], dtype=np.uint8).reshape(-1, 3, 32, 32),
                np.asarray(entry.get("labels", entry.get("fine_labels")), dtype=np.int64))

    parts = [load("data_batch_%d" % i) for i in range(1, 6)]
    tr_x = np.concatenate([p[0] for p in parts])
    tr_y = np.concatenate([p[1] for p in parts])
    te_x, te_y = load("test_batch")
    with open(base / "batches.meta", "rb") as fh:
        names = pickle.load(fh, encoding="latin1")["label_names"]
    return Dataset("cifar10",
                   Split(torch.from_numpy(np.ascontiguousarray(tr_x)), torch.from_numpy(tr_y)),
                   Split(torch.from_numpy(np.ascontiguousarray(te_x)), torch.from_numpy(te_y)),
                   10, 32, list(names))


def _stl10(root: Path) -> Dataset:
    base = root / "stl10_binary"
    if not base.is_dir():
        raise FileNotFoundError("STL-10 binaries not found under %s" % base)

    def images(name):
        a = np.fromfile(base / name, dtype=np.uint8)
        if a.size % (3 * 96 * 96):
            raise ValueError("%s: size %d is not a multiple of 3*96*96" % (name, a.size))
        return np.ascontiguousarray(a.reshape(-1, 3, 96, 96).transpose(0, 1, 3, 2))

    def labels(name):
        y = np.fromfile(base / name, dtype=np.uint8).astype(np.int64) - 1
        if y.min() < 0 or y.max() > 9:
            raise ValueError("%s: labels outside 1..10" % name)
        return y

    tr_x, tr_y = images("train_X.bin"), labels("train_y.bin")
    te_x, te_y = images("test_X.bin"), labels("test_y.bin")
    if len(tr_x) != len(tr_y) or len(te_x) != len(te_y):
        raise ValueError("STL-10 image/label counts differ")
    names_file = base / "class_names.txt"
    names = (names_file.read_text().split() if names_file.is_file()
             else [str(i) for i in range(10)])
    return Dataset("stl10", Split(torch.from_numpy(tr_x), torch.from_numpy(tr_y)),
                   Split(torch.from_numpy(te_x), torch.from_numpy(te_y)), 10, 96, names)


DATASETS = {"cifar10": _cifar10, "stl10": _stl10}


def load_dataset(cfg: DataConfig) -> Dataset:
    if cfg.name not in DATASETS:
        raise KeyError("unknown dataset %r; registered: %s" % (cfg.name, sorted(DATASETS)))
    return DATASETS[cfg.name](Path(cfg.root))


def channel_stats(images_uint8: torch.Tensor):
    x = images_uint8.to(torch.float64) / 255.0
    return (x.mean(dim=(0, 2, 3)).to(torch.float32),
            x.std(dim=(0, 2, 3), unbiased=False).to(torch.float32))


class InputPipeline:
    """``uint8 -> float32/255 -> per-channel normalisation``."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.flatten().view(1, -1, 1, 1)
        self.std = std.flatten().view(1, -1, 1, 1)

    def to(self, device) -> "InputPipeline":
        return InputPipeline(self.mean.to(device), self.std.to(device))

    def __call__(self, images_uint8: torch.Tensor) -> torch.Tensor:
        x = images_uint8.to(torch.float32) / 255.0
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)

    def describe(self) -> dict:
        return {"order": ["uint8 / 255 -> float32", "(x - mean) / std"],
                "mean": [float(v) for v in self.mean.flatten()],
                "std": [float(v) for v in self.std.flatten()]}


def build_pipeline(dataset: Dataset, cfg: DataConfig) -> InputPipeline:
    if cfg.normalization != "fit_on_train":
        raise KeyError("unknown normalization %r" % cfg.normalization)
    mean, std = channel_stats(dataset.train.images)
    for got, want, name in ((mean, cfg.expected_mean, "mean"), (std, cfg.expected_std, "std")):
        if want is not None and not np.allclose(got.numpy(), np.asarray(want), atol=cfg.stats_atol):
            raise ValueError("normalization %s %s differs from the pinned %s"
                             % (name, got.tolist(), want))
    return InputPipeline(mean, std)

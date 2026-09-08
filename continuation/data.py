"""CIFAR-10 / CIFAR-100 loading, reproducible stratified split, batching.

Design notes
------------
* Images are kept as ``uint8`` tensors on the compute device and converted to
  float in ``[0, 1]`` per batch.  The whole dataset is ~180 MB as uint8, which
  removes dataloader nondeterminism entirely.
* **No augmentation.**  No random crops, flips, rotations or colour jitter.
* Channel normalization statistics are fitted on the *training subset only*, on
  *unfiltered* images, and are shared by every run so that the normalization
  convention never varies with the transformation level.
* The transformation is applied to the raw ``[0, 1]`` float image and
  normalization is applied afterwards (see ``continuation.pipeline``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import DataConfig
from .seeding import numpy_generator

DATASETS = {
    "cifar10": {"num_classes": 10, "tv": "CIFAR10"},
    "cifar100": {"num_classes": 100, "tv": "CIFAR100"},
}


@dataclass
class Split:
    """A named subset: uint8 images ``[N,3,H,W]`` and int64 labels ``[N]``."""
    name: str
    images: torch.Tensor
    labels: torch.Tensor
    indices: np.ndarray  # indices into the original official split

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def to(self, device) -> "Split":
        return Split(self.name, self.images.to(device), self.labels.to(device), self.indices)

    def subset(self, idx, name: str) -> "Split":
        idx = np.asarray(idx)
        t = torch.as_tensor(idx, dtype=torch.long, device=self.images.device)
        return Split(name, self.images[t], self.labels[t], self.indices[idx])


@dataclass
class DatasetBundle:
    train: Split
    val: Split
    test: Split
    num_classes: int
    class_names: list
    mean: torch.Tensor  # [3] fitted on train split only, unfiltered, in [0,1] units
    std: torch.Tensor

    def to(self, device) -> "DatasetBundle":
        return DatasetBundle(
            self.train.to(device), self.val.to(device), self.test.to(device),
            self.num_classes, self.class_names,
            self.mean.to(device), self.std.to(device),
        )


def _load_raw(cfg: DataConfig):
    from torchvision import datasets

    spec = DATASETS[cfg.dataset]
    cls = getattr(datasets, spec["tv"])
    root = str(Path(cfg.root).resolve())
    tr = cls(root=root, train=True, download=cfg.download)
    te = cls(root=root, train=False, download=cfg.download)

    def pack(ds):
        arr = np.asarray(ds.data)                      # [N,H,W,3] uint8
        images = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        labels = torch.as_tensor(np.asarray(ds.targets), dtype=torch.long)
        return images, labels

    tr_x, tr_y = pack(tr)
    te_x, te_y = pack(te)
    names = list(getattr(tr, "classes", [])) or [str(i) for i in range(spec["num_classes"])]
    return tr_x, tr_y, te_x, te_y, names, spec["num_classes"]


def stratified_split(labels, num_val: int, seed: int, stratified: bool = True):
    """Return ``(train_idx, val_idx)``: sorted, disjoint, reproducible."""
    labels = np.asarray(labels)
    n = labels.shape[0]
    rng = np.random.default_rng(seed)
    if not stratified:
        perm = rng.permutation(n)
        return np.sort(perm[num_val:]), np.sort(perm[:num_val])

    classes = np.unique(labels)
    per_class = num_val // len(classes)
    remainder = num_val - per_class * len(classes)
    val_parts = []
    for i, c in enumerate(classes):
        idx = np.flatnonzero(labels == c)
        take = per_class + (1 if i < remainder else 0)
        if take > idx.size:
            raise ValueError("class %s has %d examples, cannot take %d for val"
                             % (c, idx.size, take))
        val_parts.append(rng.permutation(idx)[:take])
    val_idx = np.sort(np.concatenate(val_parts))
    mask = np.ones(n, dtype=bool)
    mask[val_idx] = False
    return np.sort(np.flatnonzero(mask)), val_idx


def channel_stats(images_uint8: torch.Tensor):
    """Per-channel mean/std in ``[0,1]`` units, computed in float64 for stability."""
    x = images_uint8.to(torch.float64) / 255.0
    mean = x.mean(dim=(0, 2, 3))
    std = x.std(dim=(0, 2, 3), unbiased=False)
    return mean.to(torch.float32), std.to(torch.float32)


def build_dataset(cfg: DataConfig) -> DatasetBundle:
    if cfg.dataset not in DATASETS:
        raise KeyError("unknown dataset %r; known: %s" % (cfg.dataset, sorted(DATASETS)))
    tr_x, tr_y, te_x, te_y, names, num_classes = _load_raw(cfg)
    train_idx, val_idx = stratified_split(
        tr_y.numpy(), cfg.num_val, cfg.split_seed, cfg.stratified
    )
    train = Split("train", tr_x[train_idx], tr_y[train_idx], train_idx)
    val = Split("val", tr_x[val_idx], tr_y[val_idx], val_idx)
    test = Split("test", te_x, te_y, np.arange(te_x.shape[0]))
    mean, std = channel_stats(train.images)   # train subset only, unfiltered
    return DatasetBundle(train, val, test, num_classes, names, mean, std)


def split_fingerprint(bundle: DatasetBundle) -> dict:
    """Small, loggable summary that pins the exact split used."""
    import hashlib

    def h(a):
        return hashlib.blake2b(np.ascontiguousarray(a, dtype=np.int64).tobytes(),
                               digest_size=8).hexdigest()

    return {
        "n_train": len(bundle.train), "n_val": len(bundle.val), "n_test": len(bundle.test),
        "train_idx_blake2b": h(bundle.train.indices),
        "val_idx_blake2b": h(bundle.val.indices),
        "val_class_counts": np.bincount(bundle.val.labels.cpu().numpy()).tolist(),
        "channel_mean": [round(float(v), 8) for v in bundle.mean],
        "channel_std": [round(float(v), 8) for v in bundle.std],
    }


def save_split(bundle: DatasetBundle, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, train_idx=bundle.train.indices, val_idx=bundle.val.indices)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(split_fingerprint(bundle), fh, indent=2)


class BatchIndexStream:
    """Deterministic minibatch index sequence, drawn from a dedicated stream.

    Epoch-wise permutations with ``drop_last=True``.  The sequence depends only
    on ``(seed, n, batch_size)`` -- never on the transformation, the model, or
    anything that might consume a different number of random draws.  Two runs
    with the same run seed therefore see *identical* sample indices in identical
    order, which is what "paired seeds" is supposed to mean here.
    """

    def __init__(self, n: int, batch_size: int, seed: int, stream: str = "batch"):
        if batch_size > n:
            raise ValueError("batch_size %d > dataset size %d" % (batch_size, n))
        self.n, self.batch_size = int(n), int(batch_size)
        self.rng = numpy_generator(seed, stream)
        self.batches_per_epoch = self.n // self.batch_size
        self._epoch = 0
        self._perm = None
        self._pos = 0

    def next_indices(self):
        if self._perm is None or self._pos >= self.batches_per_epoch:
            self._perm = self.rng.permutation(self.n)
            self._pos = 0
            self._epoch += 1
        start = self._pos * self.batch_size
        self._pos += 1
        return self._perm[start:start + self.batch_size]

    @property
    def epoch(self) -> int:
        return self._epoch

    # -- exact branch/resume support ---------------------------------------

    def state_dict(self) -> dict:
        """Everything needed to resume the *identical* index sequence.

        Saving the position (rather than only the seed) is what lets a branched
        run see exactly the same minibatch sample indices at the same global
        updates as an uninterrupted run, including when the branch point falls
        inside an epoch.
        """
        return {
            "bit_generator": self.rng.bit_generator.state,
            "perm": None if self._perm is None else np.asarray(self._perm).copy(),
            "pos": int(self._pos),
            "epoch": int(self._epoch),
            "n": self.n,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["n"]) != self.n or int(state["batch_size"]) != self.batch_size:
            raise ValueError(
                "batch stream state is for n=%s batch_size=%s but this stream is "
                "n=%s batch_size=%s" % (state["n"], state["batch_size"], self.n, self.batch_size)
            )
        self.rng.bit_generator.state = state["bit_generator"]
        self._perm = None if state["perm"] is None else np.asarray(state["perm"]).copy()
        self._pos = int(state["pos"])
        self._epoch = int(state["epoch"])


def fixed_subset_indices(n: int, size: int, seed: int, stream: str, labels=None):
    """A fixed, reproducible probe subset (class-balanced when labels are given)."""
    size = min(int(size), int(n))
    rng = numpy_generator(seed, stream)
    if labels is None:
        return np.sort(rng.permutation(n)[:size])
    labels = np.asarray(labels)
    classes = np.unique(labels)
    per = size // len(classes)
    rem = size - per * len(classes)
    parts = []
    for i, c in enumerate(classes):
        idx = np.flatnonzero(labels == c)
        take = min(per + (1 if i < rem else 0), idx.size)
        parts.append(rng.permutation(idx)[:take])
    return np.sort(np.concatenate(parts))

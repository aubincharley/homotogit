"""Split reproducibility, probe subsets, and paired minibatch index streams."""
import numpy as np
import pytest
import torch

from continuation.data import (
    BatchIndexStream,
    channel_stats,
    fixed_subset_indices,
    stratified_split,
)
from continuation.seeding import derive_seed, numpy_generator


def fake_labels(n=50000, n_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_classes, size=n)


def balanced_labels(n_per_class=5000, n_classes=10):
    return np.repeat(np.arange(n_classes), n_per_class)


def test_split_sizes_disjoint_and_covering():
    labels = balanced_labels()
    tr, va = stratified_split(labels, num_val=5000, seed=12345)
    assert tr.size == 45000 and va.size == 5000
    assert np.intersect1d(tr, va).size == 0
    assert np.union1d(tr, va).size == labels.size
    assert np.all(np.diff(tr) > 0) and np.all(np.diff(va) > 0)   # sorted


def test_split_is_stratified():
    labels = balanced_labels()
    _, va = stratified_split(labels, num_val=5000, seed=12345)
    counts = np.bincount(labels[va], minlength=10)
    assert counts.tolist() == [500] * 10


def test_split_is_reproducible_and_seed_dependent():
    labels = fake_labels()
    a = stratified_split(labels, 5000, seed=12345)
    b = stratified_split(labels, 5000, seed=12345)
    c = stratified_split(labels, 5000, seed=999)
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_split_handles_uneven_val_size():
    labels = balanced_labels()
    tr, va = stratified_split(labels, num_val=5003, seed=1)
    assert va.size == 5003 and tr.size == labels.size - 5003
    counts = np.bincount(labels[va], minlength=10)
    assert counts.max() - counts.min() == 1


def test_derived_streams_are_distinct_and_stable():
    assert derive_seed(0, "init") != derive_seed(0, "batch")
    assert derive_seed(0, "init") == derive_seed(0, "init")
    assert derive_seed(0, "init") != derive_seed(1, "init")


def test_batch_stream_is_identical_across_conditions_with_the_same_seed():
    """Paired seeds must mean identical minibatch *sample indices*, not just a
    shared seed: two conditions consuming different amounts of randomness
    elsewhere still see the same batches."""
    a = BatchIndexStream(45000, 128, seed=7)
    b = BatchIndexStream(45000, 128, seed=7)
    # simulate a condition that draws extra randomness from unrelated streams
    other = numpy_generator(7, "unrelated")
    for _ in range(100):
        other.random(1000)
        assert np.array_equal(a.next_indices(), b.next_indices())


def test_batch_stream_differs_across_seeds_and_covers_epochs():
    a = BatchIndexStream(1000, 100, seed=1)
    b = BatchIndexStream(1000, 100, seed=2)
    assert not np.array_equal(a.next_indices(), b.next_indices())
    assert a.batches_per_epoch == 10
    seen = np.concatenate([a.next_indices() for _ in range(9)])
    assert np.unique(seen).size == 900          # no repeats within an epoch


def test_batch_stream_rejects_oversized_batches():
    with pytest.raises(ValueError):
        BatchIndexStream(50, 128, seed=0)


def test_probe_subset_is_fixed_balanced_and_independent_of_run_seed():
    labels = balanced_labels(500)
    a = fixed_subset_indices(labels.size, 1000, seed=777, stream="train_probe", labels=labels)
    b = fixed_subset_indices(labels.size, 1000, seed=777, stream="train_probe", labels=labels)
    c = fixed_subset_indices(labels.size, 1000, seed=777, stream="transform_stats", labels=labels)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)             # different stream => different subset
    assert np.bincount(labels[a], minlength=10).tolist() == [100] * 10


def test_channel_stats_are_computed_on_the_given_subset_only():
    imgs = torch.zeros(4, 3, 2, 2, dtype=torch.uint8)
    imgs[:, 0] = 255
    mean, std = channel_stats(imgs)
    assert mean[0].item() == pytest.approx(1.0)
    assert mean[1].item() == pytest.approx(0.0)
    assert std[0].item() == pytest.approx(0.0, abs=1e-7)

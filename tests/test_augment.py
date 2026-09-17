"""Augmentation: the recipe, and the reproducibility the benchmark depends on."""
from __future__ import annotations

import pytest
import torch

from continuation_core import augment
from continuation_core.config import DataConfig, ExperimentConfig


def _images(n=16, c=3, h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, c, h, w), generator=g, dtype=torch.uint8)


def test_none_is_skipped_entirely_so_recorded_runs_are_untouched():
    """``build`` returns None, not an identity: the training loop skips the call."""
    assert augment.build(DataConfig()) is None
    assert ExperimentConfig().data.augmentation == "none"


def test_shape_and_dtype_survive():
    x = _images()
    out = augment.crop_flip(x, pad=4, flip_p=0.5, seed=1)
    assert out.shape == x.shape and out.dtype == x.dtype


def test_the_same_draw_identifier_gives_the_same_crop():
    """Resuming from a checkpoint must reproduce the epoch it resumes into.

    The draw is a function of *where* it happens, never of how many draws came
    before, so a resumed run sees the same crops as an uninterrupted one even
    though the ambient RNG has been advanced an unknown number of times.
    """
    x = _images()
    s = augment.batch_seed(run_seed=2, epoch=7, batch_index=3, offset=64)
    a = augment.crop_flip(x, pad=4, flip_p=0.5, seed=s)
    torch.rand(1000)                                   # disturb the ambient RNG
    b = augment.crop_flip(x, pad=4, flip_p=0.5, seed=s)
    assert torch.equal(a, b)


@pytest.mark.parametrize("field,value", [("epoch", 8), ("batch_index", 4), ("offset", 96)])
def test_a_different_position_in_training_gives_a_different_draw(field, value):
    base = dict(run_seed=2, epoch=7, batch_index=3, offset=64)
    other = dict(base, **{field: value})
    assert augment.batch_seed(**base) != augment.batch_seed(**other)


def test_two_seeds_disagree_about_the_same_images():
    x = _images()
    a = augment.crop_flip(x, pad=4, flip_p=0.5, seed=1)
    b = augment.crop_flip(x, pad=4, flip_p=0.5, seed=2)
    assert not torch.equal(a, b)


def test_zero_padding_and_no_flip_is_the_identity():
    x = _images()
    assert torch.equal(augment.crop_flip(x, pad=0, flip_p=0.0, seed=5), x)


def test_every_flip_with_no_crop_is_exactly_a_mirror():
    x = _images()
    assert torch.equal(augment.crop_flip(x, pad=0, flip_p=1.0, seed=5), x.flip(-1))


def test_about_half_the_images_are_flipped():
    x = _images(n=512)
    out = augment.crop_flip(x, pad=0, flip_p=0.5, seed=11)
    flipped = (out == x.flip(-1)).all(dim=(1, 2, 3))
    assert 0.40 < flipped.float().mean().item() < 0.60


def test_the_crop_introduces_zero_padding_and_keeps_original_pixels():
    """A shifted crop must show the zero border, and still carry real content."""
    x = torch.full((64, 3, 32, 32), 200, dtype=torch.uint8)
    out = augment.crop_flip(x, pad=4, flip_p=0.0, seed=3)
    assert (out == 0).any(), "no image was shifted off the padded edge"
    assert (out == 200).float().mean() > 0.5, "the crop lost most of the image"


@pytest.mark.parametrize("bad", [{"augmentation": "mixup"},
                                 {"augmentation": "crop_flip", "augment_pad": -1},
                                 {"augmentation": "crop_flip", "augment_flip_p": 1.5}])
def test_an_unknown_or_out_of_range_recipe_is_refused(bad):
    with pytest.raises(ValueError):
        augment.build(DataConfig(**bad))


def test_augmentation_survives_a_config_round_trip():
    cfg = ExperimentConfig()
    cfg.data = DataConfig(augmentation="crop_flip", augment_pad=4)
    assert ExperimentConfig.from_dict(cfg.to_dict()).data == cfg.data

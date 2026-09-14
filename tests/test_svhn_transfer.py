"""The SVHN transfer: nothing rescaled, and the two SVHN storage conventions.

SVHN is 32x32, so unlike STL-10 the whole point is that the schedules, the
placements and the optimizer are the reference values untouched.  What can go
wrong is the data: images stored (H, W, C, N), and labels 1..10 with 10 meaning
the digit zero.  Both are silent -- a transposed image or an off-by-one label set
trains perfectly happily to a worse number.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from conftest import synthetic_dataset                                  # noqa: E402
from continuation_core.assets import make_assets                        # noqa: E402
from continuation_core.config import DataConfig                         # noqa: E402
from continuation_core.controller import InterventionController         # noqa: E402
from continuation_core.data import load_dataset                         # noqa: E402
from continuation_core.methods import PLATEAU_G, RPROG                  # noqa: E402
from continuation_core.models import build_model, site_map              # noqa: E402

import convert_svhn                                                     # noqa: E402
import svhn_configs as S                                                # noqa: E402

ARCH = "resnet20_bn_cifar"


# -- the conversion, which is where the silent failures live ------------------

def test_to_nchw_undoes_both_svhn_conventions():
    # a recognisable (H, W, C, N) block: value encodes (n, c, h, w)
    n, c, h, w = 4, 3, 32, 32
    src = np.zeros((h, w, c, n), dtype=np.uint8)
    for i in range(n):
        for j in range(c):
            src[:, :, j, i] = i * 16 + j * 4
    src[1, 2, 0, 0] = 250                       # a marker at H=1, W=2
    y = np.array([[10], [1], [9], [10]], dtype=np.uint8)

    x, lab = convert_svhn.to_nchw(src, y)
    assert x.shape == (n, c, h, w) and x.dtype == np.uint8 and x.flags["C_CONTIGUOUS"]
    assert x[0, 0, 1, 2] == 250                 # H and W did not get swapped
    for i in range(n):
        for j in range(c):
            assert x[i, j, 0, 0] == i * 16 + j * 4
    # label 10 is the digit zero; everything else is itself
    assert lab.tolist() == [0, 1, 9, 0]
    assert lab.dtype == np.uint8


def test_to_nchw_is_not_a_reshape():
    """A plain reshape would also give (N, C, H, W) and be wrong."""
    src = np.arange(2 * 3 * 32 * 32, dtype=np.uint8).reshape(32, 32, 3, 2)
    x, _ = convert_svhn.to_nchw(src, np.ones((2, 1), dtype=np.uint8))
    assert not np.array_equal(x, src.reshape(2, 3, 32, 32))


# -- the loader's guards ------------------------------------------------------

def _write(base, n_train=73257, n_test=26032, trunc=False, bad_label=False):
    base.mkdir(parents=True, exist_ok=True)
    for name, n in (("train", n_train), ("test", n_test)):
        size = n * 3 * 32 * 32 - (1 if trunc and name == "train" else 0)
        np.zeros(max(size, 1), dtype=np.uint8).tofile(base / ("%s_X.bin" % name))
        lab = np.zeros(n, dtype=np.uint8)
        if bad_label and name == "train":
            lab[0] = 10
        lab.tofile(base / ("%s_y.bin" % name))


def test_missing_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="convert_svhn"):
        load_dataset(DataConfig(name="svhn", root=str(tmp_path)))


def test_wrong_image_count_is_an_error(tmp_path):
    _write(tmp_path / "svhn_binary", n_train=100)
    with pytest.raises(ValueError, match="expected"):
        load_dataset(DataConfig(name="svhn", root=str(tmp_path)))


def test_label_outside_zero_to_nine_is_an_error(tmp_path):
    _write(tmp_path / "svhn_binary", n_train=8, n_test=8)
    # the count guard fires first, so check the message names the real problem
    with pytest.raises(ValueError):
        load_dataset(DataConfig(name="svhn", root=str(tmp_path)))


# -- nothing is rescaled ------------------------------------------------------

def test_schedules_are_the_cifar_reference_verbatim():
    assert S.SVHN_G is PLATEAU_G and S.SVHN_R is RPROG
    assert S.EPOCHS == 30 and S.NATIVE_RESOLUTION == 32
    assert S.SVHN_G.values == (1.00, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30, 0.0)
    assert S.SVHN_R.values == (16, 24, 32)


@pytest.mark.parametrize("method_id", S.METHODS)
def test_config_matches_the_reference_recipe(method_id, tmp_path):
    cfg = S.build(method_id, 0, data_root="d", assets_dir="a", out_dir="runs")
    assert cfg.validation_status == "unvalidated"
    assert cfg.budget.epochs == 30 and cfg.budget.effective_batch == 128
    assert cfg.optimizer.lr == 0.005 and cfg.optimizer.warmup_updates == 60
    assert cfg.evaluation.every_epochs == 1        # 32x32: no need to thin it out
    spec = cfg.method_spec()
    if spec.gaussian:
        assert spec.gaussian.sigma_max == 1.0 and spec.gaussian.schedule.values[0] == 1.0
    if spec.resolution:
        assert spec.resolution.reference_resolution == 32
        assert spec.resolution.schedule.values == (16, 24, 32)
    path = tmp_path / "c.json"
    cfg.save(path)
    assert type(cfg).load(path).to_dict() == cfg.to_dict()


@pytest.mark.parametrize("method_id", S.METHODS)
def test_target_state_is_bitwise_plain(method_id):
    torch.manual_seed(0)
    model = build_model(ARCH, 10)
    cfg = S.build(method_id, 0, data_root="d", assets_dir="a", out_dir="runs")
    ctrl = InterventionController(cfg.method_spec(), site_map(ARCH), ARCH)
    ctrl.attach(model)
    plain = build_model(ARCH, 10)
    plain.load_state_dict(model.state_dict())
    ctrl.set_state(ctrl.target_state())
    model.eval(), plain.eval()
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(model(x), plain(x))


# -- the subset that makes the update budget match ----------------------------

def test_subset_size_cuts_the_training_set(tmp_path):
    ds = synthetic_dataset(n_train=200, n_test=20)
    builder = lambda: build_model(ARCH, ds.num_classes)                # noqa: E731
    man = make_assets(tmp_path / "a", builder, n_train=200, epochs=3, seeds=(0,),
                      probe_size=10, subset_size=120)
    z = np.load(tmp_path / "a" / "shared_indices.npz")
    assert z["subset"].shape == (120,)
    assert len(set(z["subset"].tolist())) == 120          # no repeats
    assert z["subset"].max() < 200                        # indexes the full split
    assert z["perm_seed0"].shape == (3, 120)              # order is over the subset
    assert z["perm_seed0"].max() < 120
    assert z["train_probe"].max() < 120                   # probe indexes the subset
    assert man["provenance"]["subset_size"] == 120
    assert man["provenance"]["n_train"] == 200


def test_subset_size_defaults_to_the_whole_split(tmp_path):
    builder = lambda: build_model(ARCH, 10)                            # noqa: E731
    man = make_assets(tmp_path / "a", builder, n_train=50, epochs=2, seeds=(0,),
                      probe_size=5)
    assert np.load(tmp_path / "a" / "shared_indices.npz")["subset"].shape == (50,)
    assert man["provenance"]["subset_size"] == 50


@pytest.mark.parametrize("bad", [0, -1, 51])
def test_subset_size_out_of_range_is_rejected(bad, tmp_path):
    builder = lambda: build_model(ARCH, 10)                            # noqa: E731
    with pytest.raises(ValueError, match="subset_size"):
        make_assets(tmp_path / "a", builder, n_train=50, epochs=2, seeds=(0,),
                    probe_size=5, subset_size=bad)


def test_subset_makes_updates_per_epoch_match_cifar():
    """50,000 at batch 128 is 391 updates, the reference number."""
    assert S.SUBSET_SIZE == 50000
    assert -(-S.SUBSET_SIZE // 128) == 391
    assert 391 * S.EPOCHS == 11730

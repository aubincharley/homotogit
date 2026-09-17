"""Checks added for the 160-epoch follow-up: schedules, augmentation, batch conventions, BN policies."""
import hashlib
import math
import random
import time

import numpy as np
import pytest
import torch

from continuation_core import assets as assets_mod
from continuation_core.augment import EpochAugmenter, apply_crop_flip, crop_flip_draws
from continuation_core.controller import InterventionController
from continuation_core.methods import CBSSpec
from continuation_core.models import build_model, site_map
from continuation_core.operators import cbs_gaussian_kernel
from continuation_core.optim import lr_at
from continuation_core.train import Trainer
from overnight import matrix as MX
from overnight.bnpolicies import POLICIES, PolicyEvaluator, bn_buffers, sha_tensors

from conftest import ROOT, synthetic_dataset


def ctrl(arm, upe=MX.UPE, total=MX.U, seed=0):
    c = InterventionController(MX.method_spec(arm), site_map("resnet20_bn_cifar"), "resnet20_bn_cifar")
    return c.configure(upe, total, seed)


# ---- recipes and schedules ---------------------------------------------------------------

def test_optimizer_schedules():
    sgd = MX.optimizer("sgd_standard_aug_160")
    assert lr_at(0, sgd, MX.U) == 0.1 and lr_at(31279, sgd, MX.U) == 0.1
    assert math.isclose(lr_at(31280, sgd, MX.U), 0.01) and math.isclose(lr_at(46919, sgd, MX.U), 0.01)
    assert math.isclose(lr_at(46920, sgd, MX.U), 0.001) and math.isclose(lr_at(MX.U - 1, sgd, MX.U), 0.001)
    assert (sgd.momentum, sgd.nesterov, sgd.weight_decay, sgd.warmup_updates) == (0.9, False, 1e-4, 0)
    adw = MX.optimizer("adamw_long_aug_160")
    assert adw == MX.optimizer("adamw_long_noaug_160")
    assert math.isclose(lr_at(0, adw, MX.U), 0.02 / 60) and lr_at(59, adw, MX.U) == 0.02 and lr_at(60, adw, MX.U) == 0.02
    assert lr_at(MX.U - 1, adw, MX.U) < 1e-8
    mid = 60 + (MX.U - 60) // 2
    assert math.isclose(lr_at(mid, adw, MX.U), 0.01, rel_tol=1e-4)
    assert MX.U == 62560 and MX.microbatch("sgd_standard_aug_160") == 128 and MX.microbatch("adamw_long_aug_160") == 32


def test_long_r_g_rg_schedules_match_update_fraction_rules():
    for u in range(MX.U):
        e = u // MX.UPE
        assert MX.R_LONG.at(e) == MX.r_by_update(u)
        assert MX.G_LONG.at(e) == MX.g_by_update(u)
    c = ctrl("resolution_max_b1_gaussian_conv")
    for u in (0, 6255, 6256, 12511, 12512, 25023, 25024, 43791, 43792, MX.U - 1):
        c.set_state(c.state_for_update(u))
        r, g = MX.r_by_update(u), MX.g_by_update(u)
        assert c.state.resolution == r and c.state.sigma == g
        assert c.per_site_sigma() == [g * r / 32.0 if g > 0 else 0.0] * 19
    trans = {a: None for a in ("resolution_max_b1", "gaussian_postrelu", "resolution_max_b1_gaussian_conv")}
    exp = {"resolution_max_b1": [12512, 25024],
           "gaussian_postrelu": [6256 * k for k in range(1, 8)],
           "resolution_max_b1_gaussian_conv": sorted({12512, 25024} | {6256 * k for k in range(1, 8)})}
    for a in trans:
        c = ctrl(a)
        got = [e * MX.UPE for e in range(1, MX.EPOCHS)
               if (c.state_for_epoch(e).resolution, c.state_for_epoch(e).sigma)
               != (c.state_for_epoch(e - 1).resolution, c.state_for_epoch(e - 1).sigma)]
        assert got == exp[a], a
    # RG keeps the upward jumps of the coupled scale at resolution transitions
    c = ctrl("resolution_max_b1_gaussian_conv")
    c.set_state(c.state_for_update(12511)); before = c.site_sigma(0)
    c.set_state(c.state_for_update(12512)); after = c.site_sigma(0)
    assert after > before                       # 0.70 * 16/32 = 0.35 -> 0.70 * 24/32 = 0.525


def test_cbs_long_schedules_integer_boundaries_and_finite_kernels():
    bm = CBSSpec(schedule="update_plateaus")
    assert bm.off_update(MX.U) == 43792 and bm.off_update(11730) == 8211 and bm.n_plateaus() == 22
    for U in (11730, MX.U):
        u_off = bm.off_update(U)
        for u in range(U):
            want = 0.0 if u >= u_off else 0.9 ** ((22 * u) // u_off)
            assert bm.sigma_at(u, 391, U) == want
        firsts = [seg["first_update"] for seg in bm.table(391, U)]
        assert firsts == [-(-k * u_off // 22) for k in range(22)] + [u_off]
    pub = CBSSpec(schedule="epoch_decay")
    for e in range(MX.EPOCHS):
        s = pub.sigma_at(e * MX.UPE, MX.UPE, MX.U)
        assert s == 0.9 ** (e // 5)
        k = cbs_gaussian_kernel(s, 1)
        assert torch.isfinite(k).all() and math.isclose(float(k.sum()), 1.0, rel_tol=1e-6)
    assert pub.sigma_at(MX.U - 1, MX.UPE, MX.U) == 0.9 ** 31
    k = cbs_gaussian_kernel(0.9 ** 31, 1)[0, 0]
    assert float(k[1, 1]) == 1.0 and float(k.sum() - k[1, 1]) == 0.0      # numerically vanishing off-centre
    c = ctrl("cbs_published_schedule")
    assert c.native_state().sigma == 0.9 ** 31


def test_long_specs_reach_exact_bypass():
    for arm in MX.ARMS:
        c = ctrl(arm)
        if arm == "cbs_published_schedule":
            continue                                   # keeps its filter by definition
        last = c.state_for_update(MX.U - 1)
        c.set_state(last if arm != "sdpoint" else c.native_state())
        assert c.is_target() and all(s == 0.0 for s in c.per_site_sigma())


def test_extended_perms_keep_the_pinned_first_30_epochs():
    z = np.load(ROOT / "assets/cifar10_resnet20bn/shared_indices.npz")
    for s in MX.SEEDS:
        base = z["perm_seed%d" % s]
        ext = MX.extended_perms(base, s)
        assert ext.shape == (160, 50000) and ext.dtype == np.int32
        assert np.array_equal(ext[:30], base)
        assert all(np.array_equal(np.sort(ext[e]), np.arange(50000)) for e in (30, 99, 159))
        assert MX.sha_array(ext) == MX.sha_array(MX.extended_perms(base, s))
        assert not np.array_equal(ext[30], ext[31])


# ---- augmentation ------------------------------------------------------------------------

def test_crop_flip_matches_explicit_slicing():
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 256, (6, 3, 32, 32), generator=g, dtype=torch.uint8)
    dx = torch.tensor([0, 8, 3, 4, 8, 0]); dy = torch.tensor([0, 8, 5, 4, 0, 8]); fl = torch.tensor([0, 1, 1, 0, 0, 1])
    out = apply_crop_flip(x, dx, dy, fl)
    padded = torch.nn.functional.pad(x, (4, 4, 4, 4))
    for i in range(6):
        want = padded[i, :, dy[i]:dy[i] + 32, dx[i]:dx[i] + 32]
        if fl[i]:
            want = want.flip(-1)
        assert torch.equal(out[i], want)
    assert torch.equal(apply_crop_flip(x, torch.full((6,), 4), torch.full((6,), 4), torch.zeros(6, dtype=torch.long)), x)


def test_draw_distribution_and_independence_from_batching():
    d = crop_flip_draws(0, 3, 90000, "augment::pad4_crop32_hflip::epoch::%d")
    for k, m in (("dx", 9), ("dy", 9), ("flip", 2)):
        counts = np.bincount(d[k], minlength=m)
        assert len(counts) == m and abs(counts / 90000 - 1 / m).max() < 0.01
    assert abs(np.corrcoef(d["dx"], d["dy"])[0, 1]) < 0.02
    d2 = crop_flip_draws(0, 3, 90000, "augment::pad4_crop32_hflip::epoch::%d")
    assert all(np.array_equal(d[k], d2[k]) for k in d)
    assert not np.array_equal(d["dx"], crop_flip_draws(0, 4, 90000, "augment::pad4_crop32_hflip::epoch::%d")["dx"])
    assert not np.array_equal(d["dx"], crop_flip_draws(1, 3, 90000, "augment::pad4_crop32_hflip::epoch::%d")["dx"])
    x = torch.randint(0, 256, (40, 3, 32, 32), dtype=torch.uint8)
    a = EpochAugmenter("pad4_crop32_hflip", 0, 40, "cpu", "augment::pad4_crop32_hflip::epoch::%d")
    idx = torch.randperm(40)
    whole = a(x[idx], idx, 2)
    parts = torch.cat([a(x[idx[i:i + 7]], idx[i:i + 7], 2) for i in range(0, 40, 7)])
    assert torch.equal(whole, parts)


def tiny(tmp_path, arm, ds, regime, epochs=3):
    cfg = MX.config(regime, arm, 0, assets_dir=str(tmp_path / "assets"), out_dir=str(tmp_path / "runs"))
    cfg.data.expected_mean = cfg.data.expected_std = None
    cfg.run.device = "cpu"
    cfg.budget.epochs = epochs
    cfg.budget.effective_batch = 32
    cfg.budget.microbatch = 32 if regime == "sgd_standard_aug_160" else 8
    cfg.evaluation.batch_size = 40
    cfg.checkpoint.at_epochs = (1, epochs)
    if cfg.optimizer.name == "sgd":
        cfg.optimizer.milestones = (5, 10)
    return cfg


@pytest.fixture
def assets(tmp_path):
    ds = synthetic_dataset(n_train=160, n_test=40)
    assets_mod.make_assets(tmp_path / "assets", lambda: build_model("resnet20_bn_cifar", 10),
                           n_train=160, epochs=6, seeds=(0,), probe_size=20)
    return ds


def test_same_augmented_examples_across_arms_and_batch_conventions(tmp_path, assets):
    seen = {}
    for regime in ("sgd_standard_aug_160", "adamw_long_aug_160"):
        for arm in ("plain", "resolution_max_b1_gaussian_conv", "sdpoint"):
            tr = Trainer(tiny(tmp_path, arm, assets, regime), dataset=assets, out_dir=tmp_path / regime / arm, log=lambda *_: None)
            rec, inner = [], tr.augmenter.__call__
            calls = {"n": 0}

            def wrap(x, idx, epoch, inner=inner, rec=rec, tr=tr, calls=calls):
                y = inner(x, idx, epoch)
                calls["n"] += 1
                rec.append((tr.global_update, idx.tolist(), hashlib.sha256(y.numpy().tobytes()).hexdigest()))
                return y
            tr.augmenter.__call__ = wrap
            tr.augmenter = type("A", (), {"__call__": staticmethod(wrap), "describe": tr.augmenter.describe})()
            tr.run(max_updates=12)
            per_update = {}
            for u, ids, h in rec:
                per_update.setdefault(u, []).append((ids, h))
            assert all(len(v) == (1 if regime.startswith("sgd") else 4) for v in per_update.values())
            flat = {u: sorted(i for ids, _ in v for i in ids) for u, v in per_update.items()}
            seen[(regime, arm)] = (flat, [h for _, _, h in rec] if regime.startswith("adamw") else None)
    ref = seen[("sgd_standard_aug_160", "plain")][0]
    assert all(v[0] == ref for v in seen.values())                     # same examples per update everywhere
    hashes = [v[1] for k, v in seen.items() if k[0] == "adamw_long_aug_160"]
    assert all(h == hashes[0] for h in hashes)                         # identical augmented pixels across arms


def test_evaluation_never_augments_and_noaug_is_identity(tmp_path, assets):
    tr = Trainer(tiny(tmp_path, "plain", assets, "adamw_long_aug_160"), dataset=assets, out_dir=tmp_path / "e", log=lambda *_: None)
    count = {"n": 0}
    orig = tr.augmenter.__call__

    class Spy:
        def __call__(self, x, idx, e):
            count["n"] += 1
            return orig(x, idx, e)
        describe = tr.augmenter.describe
    tr.augmenter = Spy()
    tr.snapshot(0, None)
    assert count["n"] == 0
    tr.run(max_updates=2)
    assert count["n"] == 8
    n0 = count["n"]
    tr.snapshot(1, None)
    assert count["n"] == n0
    none = EpochAugmenter("none", 0, 10, "cpu", "x%d")
    x = torch.zeros(2, 3, 32, 32, dtype=torch.uint8)
    assert none(x, torch.arange(2), 0) is x


@pytest.mark.parametrize("regime,arm", [("sgd_standard_aug_160", "resolution_max_b1_gaussian_conv"),
                                        ("adamw_long_aug_160", "sdpoint"),
                                        ("adamw_long_noaug_160", "cbs_budget_matched")])
def test_bitwise_resume_and_deadline_stop(tmp_path, assets, regime, arm):
    cfg = tiny(tmp_path, arm, assets, regime, epochs=3)
    straight = Trainer(cfg, dataset=assets, out_dir=tmp_path / "s", log=lambda *_: None)
    straight.run(max_updates=13)
    part = Trainer(cfg, dataset=assets, out_dir=tmp_path / "p", log=lambda *_: None)
    part.run(max_updates=6)
    assert part.run(deadline=time.time() - 1) is None                  # immediate stop: rolling.pt at update 6
    res = Trainer(cfg, dataset=assets, out_dir=tmp_path / "p", log=lambda *_: None)
    res.resume(tmp_path / "p" / "rolling.pt")
    assert res.global_update == 6
    res.run(max_updates=7)
    for a, b in zip(straight.model.state_dict().values(), res.model.state_dict().values()):
        assert torch.equal(a, b)
    for p1, p2 in zip(straight.model.parameters(), res.model.parameters()):
        for k in straight.optimizer.state[p1]:
            assert torch.equal(torch.as_tensor(straight.optimizer.state[p1][k]), torch.as_tensor(res.optimizer.state[p2][k]))
    strip = lambda ms: [{k: v for k, v in m.items() if k != "elapsed_seconds"} for m in ms]
    assert strip(straight.metrics) == strip(res.metrics)


def test_epoch_callbacks_and_checkpoints_do_not_change_training(tmp_path, assets):
    cfg = tiny(tmp_path, "resolution_max_b1_gaussian_conv", assets, "adamw_long_aug_160", epochs=3)
    a = Trainer(cfg, dataset=assets, out_dir=tmp_path / "a", log=lambda *_: None)
    a.run()
    b = Trainer(cfg, dataset=assets, out_dir=tmp_path / "b", log=lambda *_: None)
    got = []

    def cb(tr, done):
        rng = (torch.get_rng_state().clone(), np.random.get_state()[1].copy(), random.getstate())
        ev = PolicyEvaluator(tr.cfg, tr.model.state_dict(), tr.dataset, "cpu", tr.updates_per_epoch,
                             state=tr.controller.eval_current_state(done))
        got.append((done, ev.run("P1_cumulative_clean_500")))
        assert torch.equal(rng[0], torch.get_rng_state()) and np.array_equal(rng[1], np.random.get_state()[1])
        assert rng[2] == random.getstate()
    b.epoch_callbacks.append(cb)
    b.run()
    for x, y in zip(a.model.state_dict().values(), b.model.state_dict().values()):
        assert torch.equal(x, y)
    assert [d for d, _ in got] == [1, 2, 3]
    assert sorted(p.name for p in (tmp_path / "b" / "checkpoints").glob("*.pt")) == ["epoch_001.pt", "epoch_003.pt"]
    assert got[0][1]["state"]["resolution"] == 16


# ---- BN policies -------------------------------------------------------------------------

@pytest.fixture
def trained(tmp_path):
    ds = synthetic_dataset(n_train=1100, n_test=60)
    assets_mod.make_assets(tmp_path / "assets", lambda: build_model("resnet20_bn_cifar", 10),
                           n_train=1100, epochs=6, seeds=(0,), probe_size=20)
    cfg = tiny(tmp_path, "sdpoint", ds, "adamw_long_aug_160", epochs=1)
    cfg.evaluation.at_epoch_zero = False
    tr = Trainer(cfg, dataset=ds, out_dir=tmp_path / "t", log=lambda *_: None)
    tr.run()
    return ds, tr, tmp_path / "t" / "checkpoints" / "epoch_001.pt"


def test_policies_are_side_effect_free_and_p1_matches_the_frozen_panel_protocol(trained):
    ds, tr, ck = trained
    from comparison.panels import Endpoint
    ep = Endpoint(ck, ds, "cpu")
    old = ep.evaluate(ep.ctrl.native_state(), "recalibrated")
    old_saved = ep.evaluate(ep.ctrl.native_state(), "saved")
    ckd = torch.load(ck, weights_only=False)
    before = {k: v.clone() for k, v in ckd["model_state"].items()}
    pe = PolicyEvaluator(tr.cfg, ckd["model_state"], ds, "cpu", tr.updates_per_epoch)
    out = pe.all_policies()
    assert all(torch.equal(before[k], ckd["model_state"][k]) for k in before)
    P = out["policies"]
    assert P["P1_cumulative_clean_500"]["test_full"] == old["test_full"]
    assert P["P1_cumulative_clean_500"]["train_full"] == old["train_full"]
    assert P["P1_cumulative_clean_500"]["buffers_sha256"] == sha_tensors(old["buffers"])
    assert P["P0_saved_native"]["test_full"] == old_saved["test_full"]
    assert out["P0_repeat_identical"] is True
    assert P["P0_saved_native"]["buffers_sha256"] == sha_tensors(bn_buffers(before))
    assert P["P2_cumulative_clean_32"]["calibration"]["n_batches"] == 35
    assert P["P2_cumulative_clean_32"]["calibration"]["final_batch"] == 1100 - 34 * 32
    p3 = P["P3_ema_training_loader"]["calibration"]
    assert p3["batch_size"] == 8 and p3["reset"] is False and p3["momentum"] == 0.1 and p3["augmentation"] != "none"
    assert len({P[p]["buffers_sha256"] for p in POLICIES}) == 4
    assert all(P[p]["learned_tensors_unchanged"] and P[p]["state"]["sd_point"] == 0 for p in POLICIES)


def test_p3_starts_from_saved_buffers_and_is_shared_across_arms(trained):
    ds, tr, ck = trained
    ckd = torch.load(ck, weights_only=False)
    a = PolicyEvaluator(tr.cfg, ckd["model_state"], ds, "cpu", tr.updates_per_epoch).run("P3_ema_training_loader")
    zeroed = {k: (torch.zeros_like(v) if "running_mean" in k else v) for k, v in ckd["model_state"].items()}
    b = PolicyEvaluator(tr.cfg, zeroed, ds, "cpu", tr.updates_per_epoch).run("P3_ema_training_loader")
    assert a["buffers_sha256"] != b["buffers_sha256"]                 # no reset: the start matters
    noaug = PolicyEvaluator(tr.cfg, ckd["model_state"], ds, "cpu", tr.updates_per_epoch, augmentation="none").run("P3_ema_training_loader")
    assert noaug["calibration"]["order_sha256"] == a["calibration"]["order_sha256"]
    assert noaug["buffers_sha256"] != a["buffers_sha256"]

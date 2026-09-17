"""Pre-training checks of the CBS and SDPoint comparators (see comparison/METHOD_MAPPING.md)."""
import importlib.util
import math
import random
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from continuation_core import assets as assets_mod
from continuation_core.controller import InterventionController, InterventionState
from continuation_core.methods import get_method
from continuation_core.models import build_model, site_map
from continuation_core.operators import CBSGaussian, cbs_gaussian_kernel, sdpoint_size
from continuation_core.presets import reference
from continuation_core.train import Trainer

from conftest import ROOT, synthetic_dataset

REF = ROOT / "comparison" / "reference_code"
ARMS = ("plain", "resolution_max_b1", "gaussian_postrelu", "resolution_max_b1_gaussian_conv",
        "cbs_published_schedule", "cbs_budget_matched", "sdpoint")
U, UPE = 11730, 391


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def controller(method, seed=0):
    c = InterventionController(get_method(method), site_map("resnet20_bn_cifar"), "resnet20_bn_cifar")
    return c.configure(UPE, U, seed)


def model_with(method, init=None, seed=0):
    m = build_model("resnet20_bn_cifar", 10)
    if init is not None:
        m.load_state_dict(init, strict=True)
    c = controller(method, seed)
    c.attach(m)
    return m, c


# ---- pairing and identity ---------------------------------------------------------

def test_same_keys_shapes_and_initialisation_across_arms():
    init = torch.load(ROOT / "assets/cifar10_resnet20bn/init_seed0.pt", weights_only=True)
    ref = None
    for arm in ARMS:
        m = build_model("resnet20_bn_cifar", 10)                 # model construction itself draws (then overwritten)
        m.load_state_dict(init, strict=True)
        cpu_rng, np_rng, py_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
        c = controller(arm)
        c.attach(m)
        if c.cbs:                                            # building kernels must not touch any RNG
            c.set_state(InterventionState(None, 1.0))
            m.eval()
            with torch.no_grad():
                m(torch.zeros(2, 3, 32, 32))
        if c.method.sdpoint:
            c.state_for_update(0)
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert np.random.get_state()[1].tolist() == np_rng[1].tolist() and random.getstate() == py_rng
        sd = {k: v.clone() for k, v in m.state_dict().items()}
        assert sum(1 for _ in m.parameters()) == len(list(init)) - sum(1 for k in init if "running" in k or "num_batches" in k)
        if ref is None:
            ref = sd
        assert list(sd) == list(ref) and all(torch.equal(sd[k], ref[k]) for k in ref), arm
        assert assets_mod.sha_state(sd) == assets_mod.sha_state(init)


@pytest.mark.parametrize("arm,state", [
    ("cbs_published_schedule", InterventionState(None, 0.0)),
    ("cbs_budget_matched", InterventionState(None, 0.0)),
    ("sdpoint", InterventionState(None, None, sd_point=0)),
])
def test_identity_states_reproduce_plain_logits_and_gradients(arm, state):
    torch.manual_seed(0)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    base = build_model("resnet20_bn_cifar", 10)
    init = {k: v.clone() for k, v in base.state_dict().items()}
    m, c = model_with(arm, init)
    c.set_state(state)
    base.train(); m.train()
    lb, lm = base(x), m(x)
    assert torch.equal(lb, lm)
    F.cross_entropy(lb, y).backward()
    F.cross_entropy(lm, y).backward()
    for (n, a), (_, b) in zip(base.named_parameters(), m.named_parameters()):
        assert torch.equal(a.grad, b.grad), n


def test_post_add_identity_leaves_the_frozen_model_unchanged():
    torch.manual_seed(1)
    m = build_model("resnet20_bn_cifar", 10).eval()
    x = torch.randn(4, 3, 32, 32)
    out = F.relu(m.bn1(m.conv1(x)))
    for b in m.blocks:                                       # the pre-change block forward, spelled out
        h = F.relu(b.bn1(b.conv1(out)))
        out = F.relu(b.bn2(b.conv2(h)) + b.shortcut(out))
    ref = m.fc(F.adaptive_avg_pool2d(out, 1).flatten(1))
    assert torch.equal(ref, m(x))


# ---- CBS -----------------------------------------------------------------------------

def test_cbs_kernel_and_forward_match_the_author_code():
    utils = _load("cbs_utils_ref", REF / "cbs_5f62e7d" / "utils.py")
    op = CBSGaussian()
    torch.manual_seed(2)
    sigmas = sorted({get_method("cbs_published_schedule").cbs.sigma_at(u, UPE, U) for u in range(0, U, UPE)}
                    | {s for s in (get_method("cbs_budget_matched").cbs.sigma_at(u, UPE, U) for u in range(U)) if s > 0})
    assert len(sigmas) == 22              # the 6 published values coincide bitwise with budget plateaus 0-5
    for sigma in sigmas:
        for ch in (16, 32, 64):
            ref = utils.get_gaussian_filter(kernel_size=3, sigma=sigma, channels=ch)
            assert torch.equal(ref.weight.data, cbs_gaussian_kernel(sigma, ch))
            assert ref.padding == (1, 1) and ref.padding_mode == "zeros" and ref.groups == ch
            x = torch.randn(3, ch, 8, 8)
            with torch.no_grad():
                assert torch.equal(ref(x), op(x, sigma))
    # the author decays sigma by repeated multiplication; the formula 0.9**k gives the same kernels
    s_iter = 1.0
    for k in range(6):
        assert torch.equal(cbs_gaussian_kernel(s_iter, 16), cbs_gaussian_kernel(0.9 ** k, 16))
        s_iter *= 0.9


def test_cbs_sites_are_the_19_conv_outputs_before_batchnorm():
    torch.manual_seed(3)
    m, c = model_with("cbs_published_schedule")
    m.eval()
    c.set_state(InterventionState(None, 0.9))
    k = CBSGaussian()
    x = torch.randn(2, 3, 32, 32)
    conv = lambda mod, t: F.conv2d(t, mod.weight, None, mod.stride, mod.padding)   # bypasses the hooks
    out = F.relu(m.bn1(k(conv(m.conv1, x), 0.9)))
    for b in m.blocks:                                       # author BasicBlock: conv -> kernel -> bn
        h = F.relu(b.bn1(k(conv(b.conv1, out), 0.9)))
        out = F.relu(b.bn2(k(conv(b.conv2, h), 0.9)) + b.shortcut(out))
    ref = m.fc(F.adaptive_avg_pool2d(out, 1).flatten(1))
    assert torch.equal(ref, m(x))
    assert c.n_sites == 19 and len(site_map("resnet20_bn_cifar")["conv_out"]) == 19


def test_cbs_schedule_boundaries():
    pub = get_method("cbs_published_schedule").cbs
    for e in range(30):
        for u in (e * UPE, e * UPE + UPE - 1):
            assert pub.sigma_at(u, UPE, U) == 0.9 ** (e // 5)
    assert pub.sigma_at(U - 1, UPE, U) == 0.9 ** 5
    c = controller("cbs_published_schedule")
    assert c.native_state().sigma == 0.9 ** 5 and c.target_state().sigma == 0.0
    bud = get_method("cbs_budget_matched").cbs
    assert bud.n_plateaus() == 22 == math.ceil(math.log(0.1) / math.log(0.9))
    u_off = math.floor(0.7 * U)
    assert u_off == 8211
    table = bud.table(UPE, U)
    assert len(table) == 23 and table[-1] == {"first_update": 8211, "epoch": 21, "sigma": 0.0}
    for i, row in enumerate(table[:-1]):
        assert row["sigma"] == 0.9 ** i
        assert row["first_update"] == -(-i * u_off // 22)             # ceil(i * u_off / K)
        if row["first_update"] > 0:
            assert bud.sigma_at(row["first_update"] - 1, UPE, U) == 0.9 ** (i - 1)
    assert bud.sigma_at(8210, UPE, U) == 0.9 ** 21 and bud.sigma_at(U - 1, UPE, U) == 0.0
    assert controller("cbs_budget_matched").native_state().sigma == 0.0


# ---- SDPoint ------------------------------------------------------------------------

def test_author_sdpoint_selection_defects_are_real():
    res = _load("sdpoint_resnet_ref", REF / "sdpoint_0013c5d" / "models" / "resnet.py")
    net = res.resnet18()                      # BasicBlock model: blocks 0..7, stem candidate id 8
    random.seed(0)
    hits = set()
    for _ in range(400):
        net.stochastic_downsampling(None, None)
        hits |= {m.blockID for m in net.modules() if isinstance(m, res.BasicBlock) and m.downsampling_ratio < 1}
    assert hits == set()                      # defect 1: BasicBlocks are never downsampled
    chosen = []
    orig = random.randint
    try:
        random.randint = lambda a, b: 0       # force the draw of block 0
        blockID = None
        chosen.append(blockID is None and random.randint(-1, 8) or blockID)
    finally:
        random.randint = orig
    assert chosen == [None]                   # defect 2: a drawn 0 becomes None (no downsampling)


def test_sdpoint_forced_selection_reaches_every_block_with_valid_shapes():
    torch.manual_seed(4)
    m, c = model_with("sdpoint")
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    seen = {}
    sizes = {}
    for b in range(9):
        m.get_submodule("blocks.%d.post_add" % b).register_forward_hook(
            lambda _m, inp, _o, b=b: sizes.__setitem__(b, inp[0].shape[-1]))
    for p in range(1, 10):
        for r in (0.5, 0.75):
            c.set_state(InterventionState(None, None, sd_point=p, sd_ratio=r))
            m.zero_grad()
            out = m(x)
            assert out.shape == (4, 10) and torch.isfinite(out).all()
            F.cross_entropy(out, y).backward()
            assert all(torch.isfinite(q.grad).all() for q in m.parameters())
            full = {0: 32, 1: 32, 2: 32, 3: 16, 4: 16, 5: 16, 6: 8, 7: 8, 8: 8}
            assert sizes[p - 1] == full[p - 1]                # hook sees the pre-pooling map
            seen[(p, r)] = True
    assert len(seen) == 18


def test_sdpoint_block_operation_is_avgpool_after_addition_before_relu():
    torch.manual_seed(5)
    m, c = model_with("sdpoint")
    m.eval()
    x = torch.randn(2, 3, 32, 32)
    for p, r in ((1, 0.5), (4, 0.75), (9, 0.5)):
        c.set_state(InterventionState(None, None, sd_point=p, sd_ratio=r))
        out = F.relu(m.bn1(m.conv1(x)))
        for i, b in enumerate(m.blocks):
            h = F.relu(b.bn1(b.conv1(out)))
            s = b.bn2(b.conv2(h)) + b.shortcut(out)
            if i == p - 1:
                s = F.adaptive_avg_pool2d(s, int(round(s.shape[-1] * r)))
            out = F.relu(s)
        ref = m.fc(F.adaptive_avg_pool2d(out, 1).flatten(1))
        assert torch.equal(ref, m(x))
    assert [sdpoint_size(n, r) for n in (32, 16, 8) for r in (0.5, 0.75)] == [16, 24, 8, 12, 4, 6]


def test_sdpoint_draw_distribution_and_independence():
    c = controller("sdpoint", seed=0)
    draws = [c.sdpoint_draw(u) for u in range(U)]
    pts = np.bincount([p for p, _ in draws], minlength=10)
    exp = U / 10
    chi2 = ((pts - exp) ** 2 / exp).sum()
    assert chi2 < 27.88                       # chi-square 9 dof, p = 0.001
    rat = [r for _, r in draws]
    k = rat.count(0.5)
    assert abs(k - U / 2) < 3.3 * math.sqrt(U / 4)
    joint = np.zeros((10, 2))
    for p, r in draws:
        joint[p, int(r == 0.75)] += 1
    e = joint.sum(1, keepdims=True) * joint.sum(0, keepdims=True) / U
    assert ((joint - e) ** 2 / e).sum() < 27.88
    assert draws == [controller("sdpoint", seed=0).sdpoint_draw(u) for u in range(U)]
    assert draws != [controller("sdpoint", seed=1).sdpoint_draw(u) for u in range(U)]
    st = c.state_for_update(0)
    assert (st.sd_ratio is None) == (st.sd_point == 0)


# ---- trainer integration: microbatches, resume, evaluation --------------------------

def tiny(tmp_path, method, ds):
    assets_mod.make_assets(tmp_path / "assets", lambda: build_model("resnet20_bn_cifar", 10),
                           n_train=len(ds.train), epochs=4, seeds=(0,), probe_size=20)
    cfg = reference(method, 0, assets_dir=str(tmp_path / "assets"), device="cpu", out_dir=str(tmp_path / "runs"))
    cfg.data.expected_mean = cfg.data.expected_std = None
    cfg.budget.epochs, cfg.budget.effective_batch, cfg.budget.microbatch = 4, 32, 8
    cfg.evaluation.batch_size = 20
    cfg.validation_status = "test"
    return cfg


@pytest.mark.parametrize("method", ["sdpoint", "cbs_budget_matched", "cbs_published_schedule"])
def test_one_state_per_logical_batch_and_bitwise_resume(tmp_path, method):
    ds = synthetic_dataset()
    cfg = tiny(tmp_path, method, ds)
    tr = Trainer(cfg, dataset=ds, out_dir=tmp_path / "a", log=lambda *_: None)
    seen = []
    first = tr.model.get_submodule("conv1")
    first.register_forward_hook(lambda mod, *_: seen.append((tr.global_update, tr.controller.state)) if mod.training else None)
    tr.run(max_updates=9)
    key = lambda st: (st.resolution, st.sigma, st.sd_point, st.sd_ratio)       # labels are cosmetic
    by_update = {}
    for u, st in seen:
        by_update.setdefault(u, set()).add(key(st))
    assert sorted(by_update) == list(range(9))
    assert all(len(v) == 1 for v in by_update.values())            # 4 microbatches, one state
    assert all(next(iter(by_update[u])) == key(tr.controller.state_for_update(u)) for u in range(9))
    straight = Trainer(cfg, dataset=ds, out_dir=tmp_path / "s", log=lambda *_: None)
    straight.run(max_updates=15)
    part = Trainer(cfg, dataset=ds, out_dir=tmp_path / "b", log=lambda *_: None)
    part.run(max_updates=6)
    ck = part.save("rolling", name="rolling.pt")
    res = Trainer(cfg, dataset=ds, out_dir=tmp_path / "b", log=lambda *_: None)
    res.resume(ck)
    res.run(max_updates=9)
    for a, b in zip(straight.model.state_dict().values(), res.model.state_dict().values()):
        assert torch.equal(a, b)
    strip = lambda ms: [{k: v for k, v in m.items() if k != "elapsed_seconds"} for m in ms]
    assert strip(straight.metrics) == strip(res.metrics)


def test_full_tiny_run_records_native_state_and_schedule(tmp_path):
    ds = synthetic_dataset()
    cfg = tiny(tmp_path, "cbs_published_schedule", ds)
    s = Trainer(cfg, dataset=ds, out_dir=tmp_path / "r", log=lambda *_: None).run()
    assert s["native_inference_state"]["sigma"] == 1.0          # 4 epochs: still in the first 5-epoch plateau
    assert s["schedule_segments"] == [{"first_update": 0, "epoch": 0, "sigma": 1.0}]

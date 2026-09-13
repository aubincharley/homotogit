import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from continuation_core import assets as assets_mod
from continuation_core.analysis import (CheckpointEvaluator, export_trajectories,
                                        filter_normalized_direction, from_vector, pca_plane,
                                        perturbation_sensitivity, to_vector)
from continuation_core.checkpoint import load_checkpoint
from continuation_core.evaluate import evaluate
from continuation_core.models import build_model
from continuation_core.presets import reference
from continuation_core.train import Trainer

from conftest import ROOT, synthetic_dataset


def tiny_config(tmp_path, method="resolution_max_b1_gaussian_conv"):
    cfg = reference(method, 0, assets_dir=str(tmp_path / "assets"), device="cpu",
                    out_dir=str(tmp_path / "runs"))
    cfg.data.expected_mean = cfg.data.expected_std = None
    cfg.budget.epochs, cfg.budget.effective_batch, cfg.budget.microbatch = 13, 32, 16
    cfg.evaluation.batch_size = 20
    cfg.checkpoint.transition_offsets = (-1, 1)
    cfg.validation_status = "test"
    return cfg


@pytest.fixture
def setup(tmp_path):
    ds = synthetic_dataset()
    assets_mod.make_assets(tmp_path / "assets", lambda: build_model("resnet20_bn_cifar", 10),
                           n_train=len(ds.train), epochs=13, seeds=(0,), probe_size=20)
    return ds, tmp_path


def digest(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def test_assets_verify_and_detect_tampering(setup):
    ds, tmp = setup
    assert assets_mod.verify(tmp / "assets")["all_match"]
    sd = torch.load(tmp / "assets/init_seed0.pt")
    sd["fc.bias"] += 1
    torch.save(sd, tmp / "assets/init_seed0.pt")
    with pytest.raises(assets_mod.AssetMismatch):
        assets_mod.verify(tmp / "assets")


def test_resume_mid_epoch_is_bitwise(setup):
    ds, tmp = setup
    cfg = tiny_config(tmp)
    straight = Trainer(cfg, dataset=ds, out_dir=tmp / "a", log=lambda *_: None)
    straight.run(max_updates=15)                  # 5 per epoch: crosses two boundaries
    part = Trainer(cfg, dataset=ds, out_dir=tmp / "b", log=lambda *_: None)
    part.run(max_updates=7)
    ck = part.save("rolling", name="rolling.pt")
    resumed = Trainer(cfg, dataset=ds, out_dir=tmp / "b", log=lambda *_: None)
    resumed.resume(ck)
    resumed.run(max_updates=8)
    assert digest(straight.model) == digest(resumed.model)
    for p1, p2 in zip(straight.model.parameters(), resumed.model.parameters()):
        assert torch.equal(straight.optimizer.state[p1]["momentum_buffer"],
                           resumed.optimizer.state[p2]["momentum_buffer"])
    strip = lambda ms: [{k: v for k, v in m.items() if k != "elapsed_seconds"} for m in ms]
    assert json.dumps(strip(straight.metrics)) == json.dumps(strip(resumed.metrics))


def test_full_run_writes_schema_and_transition_checkpoints(setup):
    ds, tmp = setup
    cfg = tiny_config(tmp)
    tr = Trainer(cfg, dataset=ds, out_dir=tmp / "run", log=lambda *_: None)
    summary = tr.run()
    assert summary["schema"] == "continuation_core.results/1"
    assert summary["updates"] == 13 * 5
    # 13 epochs end at r=32, G=0.5: not the target state, and the summary says so
    assert summary["final_state_used"]["state"]["sigma"] == 0.5
    assert summary["final_state_is_target"] is False
    assert tr.transition_updates == [3 * 5, 6 * 5, 9 * 5, 12 * 5]   # sigma, res+sigma, sigma, res+sigma
    names = sorted(p.name for p in (tmp / "run/checkpoints").glob("update_*.pt"))
    assert names == ["update_%06d.pt" % u for u in (14, 16, 29, 31, 44, 46, 59, 61)]
    ck = load_checkpoint(tmp / "run/checkpoints/epoch_006.pt")
    used = ck["intervention"]["used_for_last_update"]["state"]
    nxt = ck["intervention"]["next_update"]["state"]
    assert (used["resolution"], used["sigma"], nxt["resolution"], nxt["sigma"]) == (16, 0.85, 24, 0.7)
    assert set(ck) >= {"model_state", "optimizer_state", "rng", "data_position", "config",
                       "lr_schedule", "global_update", "epochs_completed", "metrics"}
    rec = json.loads((tmp / "run/metrics.json").read_text())
    assert rec[0]["epoch"] == 0 and rec[-1]["epoch"] == 13
    assert set(rec[-1]["eval"]) == {"current", "target"}


def test_evaluation_has_no_side_effects_and_overrides_work(setup):
    ds, tmp = setup
    cfg = tiny_config(tmp)
    tr = Trainer(cfg, dataset=ds, out_dir=tmp / "run", log=lambda *_: None)
    tr.run(max_updates=12)
    tr.controller.set_epoch(2)
    before = (digest(tr.model), torch.get_rng_state().clone(), tr.controller.state, tr.model.training)
    x, y = tr.test_images, tr.test_labels
    for pol in ("running_stats", "fixed_batch_stats"):
        for st in (tr.controller.state_for_epoch(7), tr.controller.target_state()):
            r = evaluate(tr.model, tr.controller, tr.pipeline, x, y, st, bn_policy=pol, batch_size=20)
            assert r["bn_policy"] == pol and r["path"] in ("epoch 7", "target")
    after = (digest(tr.model), torch.get_rng_state(), tr.controller.state, tr.model.training)
    assert before[0] == after[0] and torch.equal(before[1], after[1])
    assert before[2] == after[2] and before[3] == after[3]
    st = tr.controller.target_state()
    direct = evaluate(tr.model, tr.controller, tr.pipeline, x, y, st)
    via = evaluate(tr.model, tr.controller, tr.pipeline, x, y, st,
                   params={n: p.detach().clone() for n, p in tr.model.named_parameters()})
    assert (direct["ce"], direct["acc"]) == (via["ce"], via["acc"])


def test_checkpoint_evaluator_used_next_target(setup):
    ds, tmp = setup
    cfg = tiny_config(tmp)
    Trainer(cfg, dataset=ds, out_dir=tmp / "run", log=lambda *_: None).run(max_updates=31)
    path = tmp / "run/checkpoints/epoch_006.pt"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    ev = CheckpointEvaluator(path, dataset=ds, split="train_probe", batch_size=20)
    out = {w: ev.loss(state=w) for w in ("used", "next", "target")}
    assert out["used"]["state"]["resolution"] == 16 and out["next"]["state"]["resolution"] == 24
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    res = perturbation_sensitivity(ev, [0.0, 0.1], 2, seed=0, states=("target",))
    assert res["results"][0]["rows"][0]["delta_ce_max"] == 0.0


def test_vectors_directions_and_pca(setup):
    ds, tmp = setup
    torch.manual_seed(0)
    m = build_model("resnet20_bn_cifar", 10)
    names = [n for n, _ in m.named_parameters()]
    sd = dict(m.named_parameters())
    v = to_vector(sd, names)
    back = from_vector(v, sd, names)
    assert all(torch.equal(back[n], sd[n].detach()) for n in names)
    d = filter_normalized_direction({n: p.detach() for n, p in sd.items()}, names,
                                    torch.Generator().manual_seed(0))
    w = sd["blocks.0.conv1.weight"].detach().double()
    assert torch.allclose(d["blocks.0.conv1.weight"].reshape(16, -1).norm(dim=1),
                          w.reshape(16, -1).norm(dim=1))
    assert float(d["bn1.weight"].abs().sum()) == 0.0
    g = torch.Generator().manual_seed(1)
    a, b = torch.randn(500, generator=g, dtype=torch.float64), torch.randn(500, generator=g, dtype=torch.float64)
    X = torch.stack([3 * t * a + t * t * b for t in torch.linspace(-1, 1, 7, dtype=torch.float64)])
    plane = pca_plane(X, center="mean")
    assert plane["plane_variance_ratio"] > 1 - 1e-9 and max(plane["residual_norm"]) < 1e-9


def test_trajectory_export_marks_pairing(setup):
    ds, tmp = setup
    for mid in ("plain", "resolution_max_b1"):
        cfg = tiny_config(tmp, mid)
        cfg.budget.epochs = 3
        cfg.run.name = mid
        Trainer(cfg, dataset=ds, out_dir=tmp / mid, log=lambda *_: None).run()
    meta = export_trajectories([tmp / "plain", tmp / "resolution_max_b1"], tmp / "traj.npz")
    z = np.load(tmp / "traj.npz")
    assert meta["paired"] and z["run0_vectors"].shape[0] == 3     # epoch_001..003


def test_core_imports_nothing_from_the_benchmark():
    code = ("import sys, continuation_core, continuation_core.cli, continuation_core.analysis, "
            "continuation_core.train;"
            "bad=[m for m in sys.modules if not m.startswith('torch') and "
            "(m.split('.')[0] in ('continuation','scripts') or "
            "any(k in m.lower() for k in ('wavelet','tv_budget','softpool','maxblur','perceptual',"
            "'adaptive','campaign_ops','ablation_ops','resolution_ops')))];"
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr

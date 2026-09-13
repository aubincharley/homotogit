"""Behavioural parity: continuation-core versus the executed benchmark code.

Needs a checkout of branch ``benchmark-organized`` (for the old implementation
and the recorded runs) and the CIFAR-10 python batches.  CPU, no training
campaign: a few forward/backward passes and a handful of optimizer updates.

    py verification/parity.py --benchmark-root ../benchmark-organized \
        --data-root ../../Projet_filiere/data --out verification/parity_report.json

Checks (bitwise unless stated):

 1. data arrays and normalisation statistics
 2. pinned asset digests
 3. site enumeration and reduction point
 4. schedule, resolution and effective per-site sigma at every epoch 0..31,
    against the old controller and against the ``current`` records of the
    48 unified-batch runs
 5. forward logits, loss, gradients and BN buffers at every distinct state
 6. target state == plain ResNet-20 (new) and == old target path
 7. optimizer updates across the 5->6, 11->12 and 20->21 transitions
 8. evaluation parity with the old ``evaluate`` (current and target paths)
 9. checkpoint save / load / mid-epoch resumption, and evaluation of the
    used and next states at identical weights
10. evaluation leaves parameters, buffers, RNG, controller and checkpoint intact
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT))

from continuation_core import assets as core_assets                     # noqa: E402
from continuation_core.checkpoint import load_checkpoint                 # noqa: E402
from continuation_core.config import DataConfig as CoreData             # noqa: E402
from continuation_core.controller import InterventionController         # noqa: E402
from continuation_core.data import build_pipeline, load_dataset          # noqa: E402
from continuation_core.evaluate import evaluate as core_evaluate         # noqa: E402
from continuation_core.methods import METHODS                            # noqa: E402
from continuation_core.models import build_model as core_build, site_map  # noqa: E402
from continuation_core.presets import reference                          # noqa: E402
from continuation_core.train import Trainer                              # noqa: E402

#: new method id -> unified-batch configuration id
SOURCE = {"plain": "plain", "resolution_max_b1": "shrink_b1",
          "gaussian_postrelu": "blur_relu",
          "resolution_max_b1_gaussian_conv": "shrink_b1_conv"}

REPORT: dict = {"checks": {}, "failures": []}


def record(name, ok, **details):
    REPORT["checks"][name] = {"ok": bool(ok), **details}
    if not ok:
        REPORT["failures"].append(name)
    print("%-58s %s" % (name, "OK" if ok else "FAIL %s" % details))


def same(a: torch.Tensor, b: torch.Tensor) -> tuple:
    if a.shape != b.shape:
        return False, float("inf")
    d = float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
    return bool(torch.equal(a, b)), d


def digest_model(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def git_head(root):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", required=True,
                    help="checkout of the code AS EXECUTED by the unified batch")
    ap.add_argument("--records-root",
                    help="checkout holding results/ (default: --benchmark-root)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default=str(CORE_ROOT / "verification" / "parity_report.json"))
    args = ap.parse_args()
    bench = Path(args.benchmark_root).resolve()
    records = Path(args.records_root or args.benchmark_root).resolve()
    sys.path.insert(0, str(bench))
    torch.set_num_threads(max(1, torch.get_num_threads()))
    t_start = time.perf_counter()

    from continuation.ablation_ops import build_from_cell
    from continuation.config import DataConfig as OldData, ModelConfig, OptimConfig
    from continuation.data import build_dataset
    from continuation.models import build_model as old_build
    from continuation.optim import build_optimizer as old_opt, lr_at as old_lr, set_lr as old_set_lr
    from continuation.pipeline import ChannelNormalizer, InputPipeline
    from continuation.transforms.gaussian import GaussianSmoothing
    from scripts.unified_driver import EXPECTED_STATS, evaluate as old_evaluate
    from scripts.unified_manifest import build_configs

    REPORT["environment"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                             "numpy": np.__version__, "platform": platform.platform(),
                             "device": "cpu", "threads": torch.get_num_threads(),
                             "core_commit": git_head(CORE_ROOT),
                             "executed_code_commit": git_head(bench),
                             "records_commit": git_head(records)}
    cells = {c["id"]: {**c, "seed": 0, "cell_id": c["id"] + "__seed0"} for c in build_configs()}

    # 1. data -----------------------------------------------------------------
    old_ds = build_dataset(OldData(root=args.data_root, download=False, num_val=0))
    new_ds = load_dataset(CoreData(name="cifar10", root=args.data_root))
    record("1 data: train images", torch.equal(old_ds.train.images, new_ds.train.images))
    record("1 data: train labels", torch.equal(old_ds.train.labels, new_ds.train.labels))
    record("1 data: test images", torch.equal(old_ds.test.images, new_ds.test.images))
    record("1 data: test labels", torch.equal(old_ds.test.labels, new_ds.test.labels))
    cfg0 = reference("plain", 0, data_root=args.data_root,
                     assets_dir=str(CORE_ROOT / "assets/cifar10_resnet20bn"), device="cpu")
    new_pipe = build_pipeline(new_ds, cfg0.data)
    old_pipe = InputPipeline(GaussianSmoothing(sigma_max=1.0),
                             ChannelNormalizer(old_ds.mean, old_ds.std))
    record("1 data: normalisation mean", torch.equal(new_pipe.mean.flatten(), old_ds.mean),
           value=new_pipe.mean.flatten().tolist(), pinned=EXPECTED_STATS["mean"])
    record("1 data: normalisation std", torch.equal(new_pipe.std.flatten(), old_ds.std))

    # 2. assets ---------------------------------------------------------------
    ver = core_assets.verify(CORE_ROOT / "assets/cifar10_resnet20bn")
    frozen = json.loads((records / "results/campaign_manifest_frozen.json").read_text())["assets"]
    man = json.loads((CORE_ROOT / "assets/cifar10_resnet20bn/assets_manifest.json").read_text())
    record("2 assets: all digests recomputed", ver["all_match"], n_checks=len(ver["checks"]))
    record("2 assets: manifest == campaign_manifest_frozen",
           man["states"] == frozen["states"]
           and {k: v["sha256"] for k, v in man["arrays"].items()}
           == {k: v["sha256"] for k, v in frozen["arrays"].items()})
    A = core_assets.load(CORE_ROOT / "assets/cifar10_resnet20bn", 0, verify_first=False)
    subset = torch.as_tensor(A["subset"], dtype=torch.long)
    tr_imgs, tr_lbls = new_ds.train.images[subset], new_ds.train.labels[subset]
    perms = A["perms"]
    init = A["init_state"]

    def old_model(method_id):
        m = old_build(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=0)
        m.load_state_dict(copy.deepcopy(init))
        m.train()
        ctrl, _ = build_from_cell(m, cells[SOURCE[method_id]])
        return m, ctrl

    def new_model(method_id):
        m = core_build("resnet20_bn_cifar", 10)
        m.load_state_dict(copy.deepcopy(init))
        m.train()
        ctrl = InterventionController(METHODS[method_id], site_map("resnet20_bn_cifar"))
        ctrl.attach(m)
        return m, ctrl

    # 3. sites ----------------------------------------------------------------
    m_old, _ = old_model("plain")
    names = {mod: n for n, mod in m_old.named_modules()}
    from continuation.ablation_ops import _conv_sites
    old_conv = [names[mm] for mm in _conv_sites(m_old)]
    sm = site_map("resnet20_bn_cifar")
    record("3 sites: 19 conv_out names and order", old_conv == sm["conv_out"], old=old_conv)
    old_post = [("input", "blocks")] + [("output", "blocks.%d" % b) for b in range(len(m_old.blocks))]
    record("3 sites: 10 post_relu positions", [tuple(x) for x in sm["post_relu"]] == old_post)
    from continuation.ablation_ops import INTERNAL_REDUCTIONS
    record("3 sites: block1 reduces the input of blocks.2",
           sm["reduction"]["block1"] == ("input", "blocks.%d" % INTERNAL_REDUCTIONS["block1"]["block_index"]))

    # 4. schedules --------------------------------------------------------------
    for mid in METHODS:
        _, oc = old_model(mid)
        _, nc = new_model(mid)
        rows_ok, first_bad = True, None
        for e in range(32):
            oc.set_epoch(e)
            nc.set_epoch(e)
            o_sig = [(oc.q[s] * (oc.sigma_profile[s] if oc.sigma_profile else 1.0)
                      * float(oc.value)) if (oc.is_active() and s in oc.sites) else 0.0
                     for s in range(nc.n_sites)] if nc.n_sites else []
            o_res = oc.resolution if METHODS[mid].resolution else None
            o_val = oc.value if METHODS[mid].gaussian else None
            ok = (o_res == nc.state.resolution and o_val == nc.state.sigma
                  and o_sig == nc.per_site_sigma())
            if not ok and first_bad is None:
                first_bad = {"epoch": e, "old": (o_res, o_val, o_sig),
                             "new": (nc.state.resolution, nc.state.sigma, nc.per_site_sigma())}
            rows_ok &= ok
        record("4 schedule vs old controller: %s" % mid, rows_ok, first_mismatch=first_bad)
        recs_ok, n = True, 0
        for f in glob.glob(str(records / ("results/kaggle_outputs/unified-j*/*/%s__seed*/metrics.json"
                                        % SOURCE[mid]))):
            for r in json.load(open(f)):
                k = int(r["epoch"])
                st = nc.state_for_epoch(max(k - 1, 0))
                want_r = st.resolution if st.resolution is not None else 32
                want_s = st.sigma
                recs_ok &= (r["current"]["resolution"] == want_r and r["current"]["sigma"] == want_s)
                n += 1
        record("4 schedule vs recorded runs: %s" % mid, recs_ok and n == 93, records=n)

    # 5-6. forward / backward / bypass ---------------------------------------------
    x_u8 = tr_imgs[torch.as_tensor(perms[0][:32], dtype=torch.long)]
    y = tr_lbls[torch.as_tensor(perms[0][:32], dtype=torch.long)]
    x_new = new_pipe(x_u8)
    x_old = old_pipe(x_u8, 0.0, res=32)
    record("5 pipeline output", same(x_new, x_old)[0])
    for mid in METHODS:
        epochs = sorted({min(e for e in range(30) if (METHODS[mid].resolution.schedule.at(e) if METHODS[mid].resolution else 0,
                                                       METHODS[mid].gaussian.schedule.at(e) if METHODS[mid].gaussian else 0) == key)
                         for key in {((METHODS[mid].resolution.schedule.at(e) if METHODS[mid].resolution else 0),
                                      (METHODS[mid].gaussian.schedule.at(e) if METHODS[mid].gaussian else 0))
                                     for e in range(30)}} | {29})
        worst, all_ok = 0.0, True
        for e in epochs:
            mo, oc = old_model(mid)
            mn, nc = new_model(mid)
            oc.set_epoch(e)
            nc.set_epoch(e)
            xo = old_pipe(x_u8, 0.0, res=oc.input_resolution())
            lo = F.cross_entropy(mo(xo), y)
            ln = F.cross_entropy(mn(x_new), y)
            lo.backward()
            ln.backward()
            ok_l, d = same(lo.detach(), ln.detach())
            ok = ok_l
            worst = max(worst, d)
            for (n1, p1), (n2, p2) in zip(mo.named_parameters(), mn.named_parameters()):
                okp, dp = same(p1.grad, p2.grad)
                ok &= okp and n1 == n2
                worst = max(worst, dp)
            for (n1, b1), (n2, b2) in zip(mo.named_buffers(), mn.named_buffers()):
                okb, db = same(b1, b2)
                ok &= okb
                worst = max(worst, db)
            mo.eval(), mn.eval()
            with torch.no_grad():
                oke, de = same(mo(xo), mn(x_new))
            ok &= oke
            worst = max(worst, de)
            all_ok &= ok
        record("5 forward/loss/grads/buffers: %s at epochs %s" % (mid, epochs), all_ok,
               max_abs_diff=worst)

        mn, nc = new_model(mid)
        plain = core_build("resnet20_bn_cifar", 10)
        plain.load_state_dict(copy.deepcopy(init))
        nc.set_state(nc.target_state())
        mn.eval(), plain.eval()
        mo, oc = old_model(mid)
        mo.eval()
        oc.bypass_all = True
        oc.set_state(0.0, 32)
        with torch.no_grad():
            a, b, c = mn(x_new), plain(x_new), mo(old_pipe(x_u8, 0.0, res=oc.input_resolution()))
        record("6 target state == plain ResNet-20: %s" % mid, same(a, b)[0], max_abs_diff=same(a, b)[1])
        record("6 target state == old target path: %s" % mid, same(a, c)[0])

    # 7. optimizer updates across transitions ----------------------------------
    B, MB, per_epoch = 128, 32, 391
    ocfg = OptimConfig(lr=0.005, momentum=0.9, weight_decay=5e-4, batch_size=B,
                       total_steps=30 * per_epoch, lr_schedule="cosine", warmup_steps=60,
                       min_lr=0.0)
    plan = [(0, 0), (0, 1), (5, 389), (5, 390), (6, 0), (6, 1), (11, 390), (12, 0), (20, 390), (21, 0)]
    for mid in METHODS:
        mo, oc = old_model(mid)
        opt_o = old_opt(mo, ocfg)
        cfg = reference(mid, 0, data_root=args.data_root,
                        assets_dir=str(CORE_ROOT / "assets/cifar10_resnet20bn"), device="cpu")
        tr = Trainer(cfg, dataset=new_ds, loaded_assets=A, device="cpu",
                     out_dir=tempfile.mkdtemp(prefix="parity_"), log=lambda *_: None)
        ok, worst, lrs = True, 0.0, []
        for (e, b) in plan:
            gstep = e * per_epoch + b
            oc.set_epoch(e)
            res_in = oc.input_resolution()
            batch = perms[e][b * B:b * B + B]
            idx = torch.as_tensor(batch, dtype=torch.long)
            old_set_lr(opt_o, old_lr(gstep, ocfg))
            opt_o.zero_grad(set_to_none=True)
            for a0 in range(0, int(batch.size), MB):
                sl = idx[a0:a0 + MB]
                loss = F.cross_entropy(mo(old_pipe(tr_imgs[sl], 0.0, res=res_in)), tr_lbls[sl])
                (loss * (int(sl.numel()) / int(batch.size))).backward()
            opt_o.step()
            tr.epoch, tr.batch_index, tr.global_update = e, b, gstep
            tr.controller.set_epoch(e)
            tr.train_update()
            lrs.append((opt_o.param_groups[0]["lr"], tr.optimizer.param_groups[0]["lr"]))
            for (n1, p1), (n2, p2) in zip(mo.named_parameters(), tr.model.named_parameters()):
                okp, dp = same(p1.detach(), p2.detach())
                ok &= okp
                worst = max(worst, dp)
                s1 = opt_o.state[p1].get("momentum_buffer")
                s2 = tr.optimizer.state[p2].get("momentum_buffer")
                ok &= same(s1, s2)[0]
            for (_, b1), (_, b2) in zip(mo.named_buffers(), tr.model.named_buffers()):
                ok &= same(b1, b2)[0]
        record("7 optimizer updates %s: %s" % ([p for p in plan], mid),
               ok and all(a == b for a, b in lrs), max_abs_diff=worst,
               lr_sequence=[a for a, _ in lrs])

    # 8. evaluation parity --------------------------------------------------------
    n_eval = 1000
    xe, ye = new_ds.test.images[:n_eval], new_ds.test.labels[:n_eval]
    for mid in ("resolution_max_b1_gaussian_conv", "gaussian_postrelu", "resolution_max_b1"):
        mo, oc = old_model(mid)
        mn, nc = new_model(mid)
        oce, oacc = old_evaluate(mo, oc, xe, ye, old_pipe, 6)
        tce, tacc = old_evaluate(mo, oc, xe, ye, old_pipe, 6, force_target=True)
        rn = core_evaluate(mn, nc, new_pipe, xe, ye, nc.state_for_epoch(6))
        rt = core_evaluate(mn, nc, new_pipe, xe, ye, nc.target_state())
        record("8 evaluate current+target == old evaluate: %s" % mid,
               (oce, oacc, tce, tacc) == (rn["ce"], rn["acc"], rt["ce"], rt["acc"]),
               old=(oce, oacc, tce, tacc), new=(rn["ce"], rn["acc"], rt["ce"], rt["acc"]))

    # 9. checkpoint / resume ------------------------------------------------------
    mid = "resolution_max_b1_gaussian_conv"
    cfg = reference(mid, 0, data_root=args.data_root,
                    assets_dir=str(CORE_ROOT / "assets/cifar10_resnet20bn"), device="cpu")
    cfg.evaluation.splits = ("train_probe",)
    cfg.checkpoint.transition_offsets = (-1, 1)
    start = (5, 388, 5 * per_epoch + 388)

    def fresh(out):
        t = Trainer(cfg, dataset=new_ds, loaded_assets=A, device="cpu", out_dir=out,
                    log=lambda *_: None)
        t.epoch, t.batch_index, t.global_update = start
        t.metrics = [{"placeholder": True}]            # skip the epoch-0 evaluation
        return t

    d1, d2 = tempfile.mkdtemp(prefix="straight_"), tempfile.mkdtemp(prefix="resumed_")
    straight = fresh(d1)
    straight.run(max_updates=6)
    part = fresh(d2)
    part.run(max_updates=2)                            # stops inside epoch 5
    ck = part.save("rolling", name="rolling.pt")
    resumed = fresh(d2)
    resumed.resume(ck)
    resumed.run(max_updates=4)                         # crosses the 5 -> 6 boundary
    ok = digest_model(straight.model) == digest_model(resumed.model)
    for (p1, p2) in zip(straight.model.parameters(), resumed.model.parameters()):
        ok &= same(straight.optimizer.state[p1]["momentum_buffer"],
                   resumed.optimizer.state[p2]["momentum_buffer"])[0]
    ok &= (straight.global_update, straight.epoch, straight.batch_index) == \
          (resumed.global_update, resumed.epoch, resumed.batch_index)
    record("9 mid-epoch resume == uninterrupted (6 updates across 5->6)", ok,
           final_update=straight.global_update)
    boundary = Path(d1) / "checkpoints" / "epoch_006.pt"
    cko = load_checkpoint(boundary)
    used = cko["intervention"]["used_for_last_update"]["state"]
    nxt = cko["intervention"]["next_update"]["state"]
    record("9 epoch-6 checkpoint stores used and next states separately",
           (used["resolution"], used["sigma"], nxt["resolution"], nxt["sigma"]) == (16, 0.85, 24, 0.7),
           used=used, next=nxt,
           keys=sorted(cko.keys()))
    extra = sorted(p.name for p in (Path(d1) / "checkpoints").glob("update_*.pt"))
    record("9 transition-window checkpoints written", extra == ["update_002345.pt", "update_002347.pt"],
           files=extra)

    # 10. no side effects + used/next at identical weights --------------------------
    from continuation_core.analysis import CheckpointEvaluator
    sha_before = hashlib.sha256(boundary.read_bytes()).hexdigest()
    ev = CheckpointEvaluator(boundary, dataset=new_ds, device="cpu", split="train_probe",
                             assets=A)
    rng0 = torch.get_rng_state().clone()
    md0, st0, tr0 = digest_model(ev.model), ev.controller.state, ev.model.training
    res = {}
    for which in ("used", "next", "target"):
        for pol in ("running_stats", "fixed_batch_stats"):
            r = ev.loss(state=which, bn_policy=pol)
            res["%s/%s" % (which, pol)] = {"ce": r["ce"], "acc": r["acc"],
                                           "per_site_sigma": sorted(set(r["per_site_sigma"]))}
    unchanged = (torch.equal(rng0, torch.get_rng_state()) and digest_model(ev.model) == md0
                 and ev.controller.state == st0 and ev.model.training == tr0
                 and hashlib.sha256(boundary.read_bytes()).hexdigest() == sha_before)
    record("10 evaluation leaves model, buffers, RNG, controller, file intact", unchanged)
    record("10 used vs next state at identical weights (values reported, not judged)", True,
           losses=res)

    REPORT["seconds"] = time.perf_counter() - t_start
    REPORT["all_ok"] = not REPORT["failures"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(REPORT, indent=2, default=str) + "\n", encoding="utf-8")
    print("\n%d checks, %d failures, %.0fs -> %s" % (len(REPORT["checks"]), len(REPORT["failures"]),
                                                   REPORT["seconds"], args.out))
    return 0 if REPORT["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

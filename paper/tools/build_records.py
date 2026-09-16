"""Stage 1 of the asset pipeline: original experiment records -> tidy tables in ``data/records``.

Nothing here trains or evaluates a network.  Every number is read from a committed
run record or from a raw evaluation file of the landscape study, and each output
table carries the identity of its sources.

Sources, all read-only::

    benchmark index   benchmark-organized@8d353f20 experiments/index.json, and the
                      summary.json / metrics.json of every cell it points at
    transfer runs     results/<study>/<run>/summary.json + config.json on Aubin's
                      branches (activation-transfer, groupnorm-transfer,
                      vgg11-transfer, svhn-transfer, stl10-transfer, cifar10-subset)
    optimizer runs    results/optimizer_benchmark/*_job_summary.json on
                      continuation-core-optimizer-benchmark
    input diagnostics results/kaggle_outputs/cc-probe-*/probe_shard*.json on
                      continuation-core-experiments@4179584
    landscape         studies/landscape_v{2,3} raw JSONL / Hessian JSON of the
                      visualization worktree (path from --landscape or
                      LANDSCAPE_WORKTREE; skipped, with a note, when absent)

Run from the repository that holds this file::

    py paper/tools/build_records.py [--landscape PATH]

Outputs ``data/records/*.csv`` and ``data/records/SOURCES.json`` (source commits,
row counts, sha256 of every table).  ``build_assets.py`` reads only these tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
OUT = PAPER / "data" / "records"

BENCH_COMMIT = "8d353f208661e89196ef1ab2b68b3bf4bd597d35"
INDEX_PATH = "experiments/index.json"
IDRISS_COMMIT = "4179584"
OPT_BRANCH = "origin/continuation-core-optimizer-benchmark"
TRANSFER_BRANCHES = ("activation-transfer", "groupnorm-transfer", "vgg11-transfer",
                     "svhn-transfer", "stl10-transfer", "cifar10-subset")

METHODS = ["plain", "resolution_max_b1", "gaussian_postrelu", "resolution_max_b1_gaussian_conv"]
UNIFIED_ID = {"plain": "plain", "resolution_max_b1": "shrink_b1",
              "gaussian_postrelu": "blur_relu", "resolution_max_b1_gaussian_conv": "shrink_b1_conv"}

#: Setting identity, as read from the run records themselves (see ``describe_setting``).
SETTING_ORDER = ["sgd", "adam", "adamw", "radam", "resnet20act_cifar10/gelu",
                 "resnet20act_cifar10/silu", "resnet20gn_cifar10", "vgg11_cifar10", "svhn",
                 "stl10", "cifar10_5k/stl_budget", "cifar10_5k/reference_updates"]
SETTING_LABEL = {
    "sgd": "SGD (selection batch)", "adam": "Adam", "adamw": "AdamW", "radam": "RAdam",
    "resnet20act_cifar10/gelu": "GELU", "resnet20act_cifar10/silu": "SiLU",
    "resnet20gn_cifar10": "GroupNorm", "vgg11_cifar10": "VGG-11-BN",
    "svhn": "SVHN", "stl10": "STL-10",
    "cifar10_5k/stl_budget": "CIFAR-10 5k, short budget",
    "cifar10_5k/reference_updates": "CIFAR-10 5k, reference updates"}

_cache: dict = {}


def git(*args: str) -> bytes:
    key = args
    if key not in _cache:
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)
        if r.returncode:
            raise FileNotFoundError(" ".join(args) + ": " + r.stderr.decode(errors="replace"))
        _cache[key] = r.stdout
    return _cache[key]


def show(ref: str, path: str) -> bytes:
    return git("show", "%s:%s" % (ref, path))


def show_json(ref: str, path: str):
    return json.loads(show(ref, path))


def rev(ref: str) -> str:
    return git("rev-parse", ref).decode().strip()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def write(name: str, df: pd.DataFrame, sources: dict, note: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / (name + ".csv")
    df.to_csv(p, index=False)
    print("wrote data/records/%s.csv  (%d rows)" % (name, len(df)))
    return {"rows": int(len(df)), "columns": list(df.columns), "sha256": sha_bytes(p.read_bytes()),
            "sources": sources, "note": note}


# --------------------------------------------------------------------- benchmark runs

def unified_runs() -> pd.DataFrame:
    """The 48 cells of the selection batch, from their own summary and metrics files."""
    ix = show_json(BENCH_COMMIT, INDEX_PATH)
    rows = []
    for c in ix["cells"]:
        if c["experiment"] != "unified_selected":
            continue
        s = show_json(c["summary"]["ref"] or BENCH_COMMIT, c["summary"]["path"])
        m = show_json(c["metrics"]["ref"] or BENCH_COMMIT, c["metrics"]["path"])
        last = m[-1]
        assert last["epoch"] == 30 and last["update"] == 11730 and len(m) == 31
        assert last["test_acc_current"] == last["test_acc_bypass32"] == s["final_test_acc"]
        rows.append(dict(
            setting="sgd", config_id=c["config_id"],
            method={v: k for k, v in UNIFIED_ID.items()}.get(c["config_id"], c["config_id"]),
            seed=s["seed"], dataset="cifar10", n_train=s["n_train"], n_test=s["n_test"],
            arch="resnet20_bn_cifar", activation="relu", norm="batchnorm", optimizer="sgd",
            lr=0.005, weight_decay=5e-4, epochs=s["epochs"], updates=s["updates"],
            updates_per_epoch=s["updates_per_epoch"], eval_every_epochs=1,
            test_acc=s["final_test_acc"], test_ce=s["final_test_ce"],
            train_probe_ce=s["final_train_probe_ce"], train_probe_n=500,
            last_epoch_train_loss=last["train_loss_epoch"],
            wall_seconds=s["wall_seconds"], train_seconds=s["train_seconds"],
            eval_seconds=s["eval_seconds"], device=s.get("gpu"),
            record="%s:%s" % (BENCH_COMMIT[:10], c["summary"]["path"]), batch="unified_selected"))
    return pd.DataFrame(rows)


def transfer_runs() -> pd.DataFrame:
    """Every run of Aubin's transfer studies, from summary.json + config.json."""
    rows, seen = [], set()
    for br in TRANSFER_BRANCHES:
        ref = rev("origin/" + br)
        files = git("ls-tree", "-r", "--name-only", ref).decode().split()
        for f in files:
            if not (f.startswith("results/") and f.endswith("/summary.json")):
                continue
            d = f.rsplit("/", 1)[0]
            if d.count("/") < 2 or (d + "/config.json") not in files:
                continue
            s = show_json(ref, f)
            if "method_definition" not in s:
                continue
            c = show_json(ref, d + "/config.json")
            study = "/".join(d.split("/")[1:-1])
            key = (study, s["method"], s["seed"])
            if key in seen:
                continue
            seen.add(key)
            fin = s["final"]
            path = "target" if "target" in fin else "current"
            opts = c["model"].get("options") or {}
            rows.append(dict(
                setting=study, config_id=s["method"], method=s["method"].replace("__transfer", ""),
                seed=s["seed"], dataset=s["dataset"], n_train=s["n_train"], n_test=s["n_test"],
                arch=s["arch"], activation=opts.get("activation") or "relu",
                norm="groupnorm(%d)" % opts["groups"] if opts.get("groups") else "batchnorm",
                optimizer=c["optimizer"]["name"], lr=c["optimizer"]["lr"],
                weight_decay=c["optimizer"]["weight_decay"], epochs=s["epochs"],
                updates=s["updates"], updates_per_epoch=s["updates_per_epoch"],
                eval_every_epochs=c["evaluation"]["every_epochs"],
                test_acc=fin[path]["test"]["acc"], test_ce=fin[path]["test"]["ce"],
                train_probe_ce=fin[path]["train_probe"]["ce"],
                train_probe_acc=fin[path]["train_probe"]["acc"],
                train_probe_n=fin[path]["train_probe"]["n"], endpoint_path=path,
                wall_seconds=s["timing"]["wall_seconds"], train_seconds=s["timing"]["train_seconds"],
                eval_seconds=s["timing"]["eval_seconds"], device=None,
                subset_sha=s["provenance"]["asset_arrays"]["subset"],
                probe_sha=s["provenance"]["asset_arrays"]["train_probe"],
                init_sha=s["provenance"]["asset_states"]["init_seed%d" % s["seed"]],
                order_sha=s["provenance"]["asset_arrays"]["perm_seed%d" % s["seed"]],
                code=s["provenance"].get("code"), record="%s:%s" % (ref[:10], f),
                batch="transfer"))
    return pd.DataFrame(rows)


def optimizer_runs() -> pd.DataFrame:
    """The optimizer campaign, from the job summaries that its kernels wrote."""
    ref = rev(OPT_BRANCH)
    rows = []
    for job in ("grid_0", "grid_1", "grid_2", "lr_sweep", "lr_sweep_ext", "sgd_control"):
        d = show_json(ref, "results/optimizer_benchmark/%s_job_summary.json" % job)
        for r in d["rows"]:
            rows.append(dict(
                setting=r["optimizer"] if d["batch"] == "grid" else d["batch"],
                config_id=r["run"], method=r["method"], seed=r["seed"], dataset="cifar10",
                n_train=50000, n_test=10000, arch="resnet20_bn_cifar", activation="relu",
                norm="batchnorm", optimizer=r["optimizer"], lr=r["lr"], weight_decay=5e-4,
                epochs=30, updates=11730, updates_per_epoch=391, eval_every_epochs=1,
                test_acc=r["final_test_acc"], wall_seconds=r["wall_seconds"],
                status=r["status"], record="%s:results/optimizer_benchmark/%s_job_summary.json"
                % (ref[:10], job), batch=d["batch"]))
    return pd.DataFrame(rows)


def describe_setting(g: pd.DataFrame) -> dict:
    """Dataset, size, model and budget of one setting, taken from its own runs."""
    one = lambda col: sorted({x for x in g[col].dropna().tolist()})  # noqa: E731
    vals = {c: one(c) for c in ("dataset", "n_train", "n_test", "arch", "activation", "norm",
                                "optimizer", "lr", "epochs", "updates", "updates_per_epoch",
                                "eval_every_epochs", "train_probe_n")}
    for k, v in vals.items():
        assert len(v) <= 1, (k, v)
    return {k: (v[0] if v else None) for k, v in vals.items()}


def benchmark_tables():
    u, t, o = unified_runs(), transfer_runs(), optimizer_runs()
    runs = pd.concat([u, t, o[o.batch == "grid"]], ignore_index=True)
    runs = runs[runs.method.isin(METHODS)].copy()
    pilots = o[o.batch != "grid"].copy()
    rows = []
    for setting in SETTING_ORDER:
        g = runs[runs.setting == setting]
        assert len(g) == 12, (setting, len(g))
        d = describe_setting(g)
        base = g[g.method == "plain"].set_index("seed")
        for m in METHODS:
            z = g[g.method == m].set_index("seed").sort_index()
            diff = 100 * (z.test_acc - base.test_acc)
            rows.append(dict(
                setting=setting, label=SETTING_LABEL[setting], method=m, n_seeds=len(z),
                seeds=",".join(str(s) for s in z.index), **d,
                test_acc_mean=100 * z.test_acc.mean(), test_acc_sd=100 * z.test_acc.std(),
                test_acc_per_seed=",".join("%.2f" % (100 * v) for v in z.test_acc),
                gain_mean=diff.mean(), gain_sd=diff.std(),
                gain_per_seed=",".join("%+.2f" % v for v in diff),
                n_seeds_gain_positive=int((diff > 0).sum()),
                test_ce_mean=z.test_ce.mean() if z.test_ce.notna().all() else None,
                test_ce_sd=z.test_ce.std() if z.test_ce.notna().all() else None,
                train_probe_ce_mean=z.train_probe_ce.mean() if "train_probe_ce" in z and z.train_probe_ce.notna().all() else None,
                train_probe_ce_sd=z.train_probe_ce.std() if "train_probe_ce" in z and z.train_probe_ce.notna().all() else None,
                wall_minutes_mean=z.wall_seconds.mean() / 60,
                wall_minutes_sd=z.wall_seconds.std() / 60,
                wall_ratio_to_plain_mean=(z.wall_seconds / base.wall_seconds).mean(),
                wall_ratio_to_plain_sd=(z.wall_seconds / base.wall_seconds).std()))
    settings = pd.DataFrame(rows)
    return runs, pilots, settings


# --------------------------------------------------------------------- historical catalogue

def catalogue_tables():
    ix = show_json(BENCH_COMMIT, INDEX_PATH)
    exps = {e["id"]: e for e in ix["experiments"]}
    cells = []
    for c in ix["cells"]:
        s = show_json(c["summary"]["ref"] or BENCH_COMMIT, c["summary"]["path"])
        acc = s.get("final_test_acc", s.get("final_val_acc"))
        ce = s.get("final_test_ce", s.get("final_val_ce"))
        if isinstance(s.get("final"), dict) and "target_val" in s["final"]:
            acc, ce = s["final"]["target_val"].get("accuracy"), s["final"]["target_val"].get("ce")
        if "full_val_acc_unfiltered" in s:
            acc, ce = s["full_val_acc_unfiltered"], s["full_val_ce_unfiltered"]
        if c["experiment"] == "ablation_aa" and s.get("final_test_acc_primary") is not None:
            acc = s["final_test_acc_primary"]
            ce = (s.get("final_test_ce_current") if s.get("primary_path") == "current"
                  else s.get("final_test_ce"))
        cells.append(dict(experiment=c["experiment"], config_id=c["config_id"],
                          cell_id=c["cell_id"], seed=c["seed"], group=c["group"],
                          acc_from_record=acc, ce_from_record=ce,
                          valid=bool(finite(ce) if ce is not None else True),
                          wall_seconds=s.get("wall_seconds"),
                          index_final_acc=c["final_acc"], index_valid=c["numerically_valid"],
                          record="%s:%s" % ((c["summary"]["ref"] or BENCH_COMMIT)[:10],
                                            c["summary"]["path"])))
    cells = pd.DataFrame(cells)
    rows = []
    for cf in ix["configurations"]:
        groups = set(cf["run_conditions"].get("groups") or [])
        mem = cells[(cells.experiment == cf["experiment"]) & (cells.config_id == cf["config_id"])]
        if groups:
            mem = mem[mem.group.isin(groups)]
        v = mem[mem.valid & mem.acc_from_record.notna()]
        e = exps[cf["experiment"]]
        walls = v.wall_seconds.dropna()
        rows.append(dict(
            experiment=cf["experiment"], experiment_title=e["title"], owner=e["owner"],
            conditions=e["conditions"], evaluation_split=cf.get("split"),
            timing_scope=e["timing_scope"], config_id=cf["config_id"], label=cf["label"],
            family=cf.get("family"), status=cf["status"], flags=";".join(cf["flags"]),
            seeds_attempted=len(cf["seeds_attempted"]), seeds_valid=len(v),
            acc_mean=100 * v.acc_from_record.mean() if len(v) else None,
            acc_sd=100 * v.acc_from_record.std() if len(v) > 1 else None,
            acc_per_seed=",".join("%.2f" % (100 * a) for a in v.acc_from_record),
            wall_seconds_mean=walls.mean() if len(walls) else None,
            final_path=cf["final_path"] if isinstance(cf["final_path"], str) else
            ", ".join(cf["final_path"]),
            asset_sets=";".join(cf["asset_sets"]), epochs=cf["run_conditions"].get("epochs"),
            n_train=cf["run_conditions"].get("n_train"),
            index_acc_mean=None if cf["acc_mean"] is None else 100 * cf["acc_mean"],
            index_acc_sd=None if cf["acc_sd"] is None else 100 * cf["acc_sd"]))
    return cells, pd.DataFrame(rows)


# --------------------------------------------------------------------- input diagnostics

def idriss_tables():
    names = git("ls-tree", "-r", "--name-only", IDRISS_COMMIT).decode().split()
    cells, freq = [], []
    for n in names:
        if "probe_shard" not in n or not n.endswith(".json"):
            continue
        for cell, v in show_json(IDRISS_COMMIT, n)["cells"].items():
            sen, cur = v["sensitivity"], v["curvature"]
            method = cell.rsplit("__", 2)[0]
            row = dict(cell=cell, method=method, recipe_variant=cell.split("__")[-2],
                       training_seed=int(cell[-1]), train_n=v["train"]["n"],
                       train_ce=v["train"]["ce"], train_error=v["train"]["err"],
                       test_n=v["test"]["n"], test_ce=v["test"]["ce"], test_error=v["test"]["err"],
                       jac_images=sen["n_images"], jac_frobenius=sen["jacobian"]["frobenius_mean"],
                       jac_spectral=sen["jacobian"]["spectral_mean"],
                       jac_participation=sen["jacobian"]["participation_mean"],
                       freq_mean_radius=sen["frequency"]["mean_radius"],
                       hess_images=cur["n_images"], hess_parameters=cur["n_parameters"],
                       hutchinson_draws=cur["draws"], trace_deflated=cur["trace"]["mean"],
                       trace_raw=cur["trace_raw"]["mean"], top5_sum=cur["trace"]["eigen_sum"],
                       top_eig1=cur["top_eigenvalues"][0]["eigenvalue"],
                       top5_converged=sum(e["converged"] for e in cur["top_eigenvalues"]),
                       record="%s:%s" % (IDRISS_COMMIT, n))
            for a in sen["amplitude_curve"]:
                row["logit_displacement_eps%g" % a["epsilon"]] = a["displacement"]
            for a in sen["linearity_ratio"]:
                row["linearity_ratio_eps%g" % a["epsilon"]] = a["ratio"]
            cells.append(row)
            for i, f in enumerate(sen["frequency"]["ring_energy_fraction"]):
                freq.append(dict(cell=cell, method=method,
                                 recipe_variant=cell.split("__")[-2], radius=i, energy_fraction=f))
    return pd.DataFrame(cells), pd.DataFrame(freq)


# --------------------------------------------------------------------- landscape study

def landscape_tables(worktree: Path):
    sys.path.insert(0, str(worktree))
    from landscape_v2 import common as V2  # noqa: E402
    from landscape_v3 import common as C, v2points  # noqa: E402

    points, source = {}, {}
    dirs = sorted(p for p in (C.STUDY / "raw").glob("*/v3")
                  if p.is_dir() and not p.parent.name.endswith("_pilot"))
    for d in dirs:
        for f in sorted((d / "eval").glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if "splits" in r and "|" in str(r.get("key", "")) and r["key"] not in points:
                    points[r["key"]] = r
                    source[r["key"]] = "%s/%s" % (d.parent.name, f.name)
    v2 = v2points.load()
    for k, r in v2.items():
        points.setdefault(k, r)
        source.setdefault(k, r["source"])
    idx = defaultdict(dict)
    for k, r in points.items():
        m, s, ck, stg, pol, prec, ss, spec = k.split("|")
        idx[(C.LONG[m], int(s[1:]), ck, stg, pol, prec, ss)][spec] = r

    def final_key(m, s, pol, ss, prec="f32"):
        return (m, s, "epoch_030", C.stag(C.target(m)), pol, prec, ss)

    # per-direction sensitivity at the final checkpoints
    rows = []
    for m in C.METHODS:
        for s in C.SEEDS:
            for pol in C.POLICIES:
                pts = idx.get(final_key(m, s, pol, "probes"), {})
                if "c" not in pts:
                    continue
                for split in C.PROBES:
                    L0 = pts["c"]["splits"][split]["ce"]
                    for k in range(C.N_DIRECTIONS):
                        for a in C.AMPLITUDES:
                            p, q = pts.get(C.spec_r1(k, a)), pts.get(C.spec_r1(k, -a))
                            if p is None or q is None:
                                continue
                            dp = p["splits"][split]["ce"] - L0
                            dm = q["splits"][split]["ce"] - L0
                            rows.append(dict(method=m, seed=s, policy=pol, split=split,
                                             direction=k, amplitude=a, centre_ce=L0,
                                             dL_plus=dp, dL_minus=dm, S=(dp + dm) / 2))
    per_dir = pd.DataFrame(rows)
    per_seed = (per_dir.groupby(["method", "seed", "policy", "split", "amplitude"])
                .S.agg(S_mean="mean", S_direction_sd="std", n_directions="size").reset_index())
    base = per_seed[per_seed.method == "plain"].set_index(
        ["seed", "policy", "split", "amplitude"]).S_mean
    oth = per_seed[per_seed.method != "plain"].copy()
    oth["delta"] = [r.S_mean - base.loc[(r.seed, r.policy, r.split, r.amplitude)]
                    for r in oth.itertuples()]
    paired = (oth.groupby(["method", "policy", "split", "amplitude"]).delta
              .agg(mean_delta="mean", sd_delta="std", n_seeds="size",
                   n_seeds_below_plain=lambda v: int((v < 0).sum())).reset_index())

    # larger evaluation sets
    rows = []
    for m in C.METHODS:
        for s in C.SEEDS:
            for pol in (C.SAVED, C.POINTWISE):
                pts = idx.get(final_key(m, s, pol, "large"), {})
                if "c" not in pts:
                    continue
                for split in C.LARGE:
                    L0 = pts["c"]["splits"][split]["ce"]
                    for a in C.VALIDATION_AMPS:
                        vals = [(pts[C.spec_r1(k, a)]["splits"][split]["ce"]
                                 + pts[C.spec_r1(k, -a)]["splits"][split]["ce"]) / 2 - L0
                                for k in range(C.N_DIRECTIONS)
                                if C.spec_r1(k, a) in pts and C.spec_r1(k, -a) in pts]
                        rows.append(dict(method=m, seed=s, policy=pol, evaluation_set=split,
                                         n_images=pts["c"]["splits"][split]["n"], amplitude=a,
                                         n_directions=len(vals), S=float(np.mean(vals)),
                                         centre_ce=L0,
                                         centre_acc=pts["c"]["splits"][split]["acc"]))
    large = pd.DataFrame(rows)
    b = large[large.method == "plain"].set_index(
        ["seed", "policy", "evaluation_set", "amplitude"]).S
    o = large[large.method != "plain"].copy()
    o["delta"] = [r.S - b.loc[(r.seed, r.policy, r.evaluation_set, r.amplitude)]
                  for r in o.itertuples()]
    large_paired = (o.groupby(["method", "policy", "evaluation_set", "amplitude"]).delta
                    .agg(mean_delta="mean", sd_delta="std", n_seeds="size",
                         n_seeds_below_plain=lambda v: int((v < 0).sum())).reset_index())

    # endpoints of the twenty landscape models
    rows = []
    for m in C.METHODS:
        for s in C.SEEDS:
            pts = idx.get(final_key(m, s, C.SAVED, "large"), {})
            sm = json.loads((C.v2_run_dir(m, s) / "summary.json").read_text())
            fin = sm["final"]["target" if "target" in sm["final"] else "current"]
            sp = pts["c"]["splits"]
            rows.append(dict(method=m, seed=s, policy=C.SAVED,
                             train_full_n=sp["train_full"]["n"], train_full_ce=sp["train_full"]["ce"],
                             train_full_acc=sp["train_full"]["acc"],
                             test_full_n=sp["test_full"]["n"], test_full_ce=sp["test_full"]["ce"],
                             test_full_acc=sp["test_full"]["acc"],
                             training_log_test_acc=fin["test"]["acc"],
                             training_log_test_ce=fin["test"]["ce"],
                             wall_seconds=sm["timing"]["wall_seconds"]))
    endpoints = pd.DataFrame(rows)

    # Hessian eigenproblems
    rows = []
    for d in dirs:
        for f in sorted((d / "hessian").glob("*.json")):
            h = json.loads(f.read_text())
            t, s0, s1 = h["task"], h["starts"]["0"], h["starts"]["1"]
            rows.append(dict(problem=h["problem"], method=t["m"], seed=t["s"], probe=t["probe"],
                             policy=t["policy"], coords=t["coords"], loss=h["loss"],
                             grad_norm=h["grad_norm"], n_params=h["n_params"],
                             n_blocks=h["block_info"]["n_nonzero_blocks"],
                             top1=s0["values"]["top1"], top2=s0["values"]["top2"],
                             min=s0["values"]["min"],
                             top1_rel_resid=s0["residuals"]["top1"]["rel_resid"],
                             top2_rel_resid=s0["residuals"]["top2"]["rel_resid"],
                             min_rel_resid=s0["residuals"]["min"]["rel_resid"],
                             top1_start1=s1["values"]["top1"], min_start1=s1["values"]["min"],
                             converged_start0=all(s0["converged_explicit"].values()),
                             converged_start1=all(s1["converged_explicit"].values()),
                             iterations_start0=s0["iterations"], iterations_start1=s1["iterations"],
                             min_rayleigh_f64=h["verification"]["min"]["rayleigh_f64_coordinates"],
                             source=d.parent.name))
    hess = pd.DataFrame(rows).drop_duplicates("problem")

    # numerical checks: repeated/symmetric HVPs and the float64 recomputation
    checks = []
    for d in dirs:
        for f in sorted((d / "eval").glob("hchecks__*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    checks.append(json.loads(line))
    hvp = pd.DataFrame(checks).drop_duplicates(["m", "s", "policy", "probe"])
    rows = []
    for m in C.METHODS:
        for s in C.SEEDS:
            for pol in C.POLICIES:
                p64 = idx.get(final_key(m, s, pol, "probes", "f64"), {})
                p32 = idx.get(final_key(m, s, pol, "probes"), {})
                if "c" not in p64:
                    continue
                for split in C.PROBES:
                    for k in C.F64_CHECK["directions"]:
                        for a in C.F64_CHECK["amplitudes"]:
                            kp, km = C.spec_r1(k, a), C.spec_r1(k, -a)
                            if not {kp, km} <= set(p64) or not {kp, km} <= set(p32):
                                continue
                            S64 = ((p64[kp]["splits"][split]["ce"] + p64[km]["splits"][split]["ce"])
                                   / 2 - p64["c"]["splits"][split]["ce"])
                            S32 = ((p32[kp]["splits"][split]["ce"] + p32[km]["splits"][split]["ce"])
                                   / 2 - p32["c"]["splits"][split]["ce"])
                            rows.append(dict(method=m, seed=s, policy=pol, split=split,
                                             direction=k, amplitude=a, S_f32=S32, S_f64=S64,
                                             abs_diff=abs(S64 - S32)))
    f64 = pd.DataFrame(rows)

    # fixed weights at every schedule transition
    rows = []
    for m in C.METHODS[1:]:
        ctrl = V2.controller(m)
        for tr in V2.transitions(m):
            states = {"before": tr["before"], "after": tr["after"],
                      "final": V2.state_dict_of(ctrl.target_state())}
            for label, stt in states.items():
                for s in C.SEEDS:
                    for pol in (C.SAVED, C.POINTWISE):
                        pts = idx.get((m, s, "epoch_%03d" % tr["epoch"], C.stag(stt), pol,
                                       "f32", "probes"), {})
                        if "c" not in pts:
                            continue
                        for split in C.PROBES:
                            rows.append(dict(method=m, epoch=tr["epoch"], state_label=label,
                                             state=C.stag(stt), seed=s, policy=pol, split=split,
                                             centre_ce=pts["c"]["splits"][split]["ce"]))
    trans = pd.DataFrame(rows)
    rows = []
    for (m, e, split, pol), g in trans.groupby(["method", "epoch", "split", "policy"]):
        w = g.pivot_table(index="seed", columns="state_label", values="centre_ce")
        if not {"before", "after"} <= set(w.columns):
            continue
        rows.append(dict(method=m, epoch=e, split=split, policy=pol, n_seeds=len(w),
                         n_states_compared=len(w.columns),
                         n_seeds_trained_state_lowest=int((w["before"] <= w.min(axis=1) + 1e-12).sum()),
                         n_seeds_next_state_lower=int((w["after"] < w["before"]).sum())))
    trans_audit = pd.DataFrame(rows)

    # straight segments between solutions
    rows = []
    for s in C.SEEDS:
        ev = C.V2_RAW / C.V2_ACCOUNT_OF_SEED[s] / "v2" / "eval"
        for f in sorted(ev.glob("interp__*__seed%d.jsonl" % s)):
            pair = f.name.split("__")[1]
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows.append(dict(pair=pair, seed=s, alpha=r["alpha"], policy=r["policy"],
                                     train_probe_ce=r["splits"]["train_probe"]["ce"],
                                     test_probe_ce=r["splits"]["test_probe"]["ce"],
                                     source="v2:" + f.name))
    interp = pd.DataFrame(rows)
    rows = []
    for (pair, s), g in interp.groupby(["pair", "seed"]):
        g = g.sort_values("alpha")
        rec = dict(pair=pair, seed=s, n_points=len(g))
        for split in ("train_probe", "test_probe"):
            v = g[split + "_ce"].to_numpy()
            rec[split + "_barrier"] = float(v.max() - max(v[0], v[-1]))
            rec[split + "_argmax_alpha"] = float(g.alpha.to_numpy()[v.argmax()])
        rows.append(rec)
    barriers = pd.DataFrame(rows)

    # PCA of the recorded trajectories
    rows = []
    for s in C.SEEDS:
        z = np.load(C.V2_RAW / C.V2_ACCOUNT_OF_SEED[s] / "v2" / "pca" / ("pca_seed%d.npz" % s))
        evr = np.cumsum(z["explained_variance_ratio"])
        frac = {k: (z["resid%d" % k] / z["dist_to_mean"]) ** 2 for k in (2, 5, 10)}
        rows.append(dict(seed=s, n_checkpoints=len(z["dist_to_mean"]),
                         evr_pc1=float(z["explained_variance_ratio"][0]), evr_pc1_2=float(evr[1]),
                         evr_pc1_4=float(evr[3]), evr_pc1_10=float(evr[9]),
                         median_unexplained_2pc=float(np.median(frac[2])),
                         max_unexplained_2pc=float(frac[2].max()),
                         median_unexplained_5pc=float(np.median(frac[5])),
                         max_unexplained_5pc=float(frac[5].max())))
    pca = pd.DataFrame(rows)

    # the wide random planes, vertex by vertex
    rows = []
    for m in C.METHODS:
        for s in C.SEEDS:
            pts = idx.get(final_key(m, s, C.POINTWISE, "probes"), {})
            grid = C.WIDE41 if s == C.PRIMARY_SEED else C.WIDE21
            for a in grid:
                for bb in grid:
                    r = pts.get(C.spec_r2(0, 1, a, bb)) or (pts.get("c") if a == 0 and bb == 0
                                                            else None)
                    rows.append(dict(method=m, seed=s, grid=len(grid), a=a, b=bb,
                                     train_probe_ce=None if r is None else r["splits"]["train_probe"]["ce"],
                                     test_probe_ce=None if r is None else r["splits"]["test_probe"]["ce"],
                                     source=None if r is None else source.get(
                                         C.pkey(m, s, C.FINAL, C.target(m), C.POINTWISE, "probes",
                                                C.spec_r2(0, 1, a, bb)))))
    planes = pd.DataFrame(rows)

    stats = dict(n_v3_points=len(points) - len(v2), n_v2_points=len(v2),
                 accounts=[d.parent.name for d in dirs])
    return dict(landscape_sensitivity_per_direction=per_dir,
                landscape_sensitivity_per_seed=per_seed,
                landscape_sensitivity_paired=paired,
                landscape_large_set_per_seed=large,
                landscape_large_set_paired=large_paired,
                landscape_endpoints=endpoints, landscape_hessian=hess,
                landscape_hvp_checks=hvp, landscape_float64_checks=f64,
                landscape_transitions=trans, landscape_transitions_audit=trans_audit,
                landscape_interpolation=interp, landscape_interpolation_barriers=barriers,
                landscape_pca=pca, landscape_random_planes=planes), stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--landscape", default=os.environ.get("LANDSCAPE_WORKTREE",
                                                          str(ROOT.parent / "visualization")),
                    help="worktree holding studies/landscape_v2 and _v3 raw outputs")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"generator": "paper/tools/build_records.py",
                "benchmark_index": "%s:%s" % (BENCH_COMMIT[:10], INDEX_PATH),
                "idriss_commit": rev(IDRISS_COMMIT), "optimizer_branch": rev(OPT_BRANCH),
                "transfer_branches": {b: rev("origin/" + b) for b in TRANSFER_BRANCHES},
                "tables": {}}

    runs, pilots, settings = benchmark_tables()
    manifest["tables"]["benchmark_runs"] = write(
        "benchmark_runs", runs, {"index": manifest["benchmark_index"]},
        "one row per training run of the twelve reported settings")
    manifest["tables"]["benchmark_pilot_runs"] = write(
        "benchmark_pilot_runs", pilots, {"branch": manifest["optimizer_branch"]},
        "learning-rate pilots and the SGD drift control; not part of any reported setting")
    manifest["tables"]["benchmark_settings"] = write(
        "benchmark_settings", settings, {"derived_from": "benchmark_runs"},
        "per setting and method: absolute accuracy, paired gain, CE and wall time")

    cells, configs = catalogue_tables()
    manifest["tables"]["catalogue_cells"] = write(
        "catalogue_cells", cells, {"index": manifest["benchmark_index"]},
        "every historical run, with the value recomputed from its own summary record")
    manifest["tables"]["catalogue_configurations"] = write(
        "catalogue_configurations", configs, {"index": manifest["benchmark_index"]},
        "the 165 historical configurations, aggregated from the cell records")

    icells, ifreq = idriss_tables()
    manifest["tables"]["input_diagnostics_cells"] = write(
        "input_diagnostics_cells", icells, {"commit": manifest["idriss_commit"]},
        "the 28 input-derivative grid cells")
    manifest["tables"]["input_diagnostics_frequency"] = write(
        "input_diagnostics_frequency", ifreq, {"commit": manifest["idriss_commit"]},
        "Jacobian energy by spatial frequency radius")

    w = Path(a.landscape)
    if (w / "studies" / "landscape_v3" / "raw").is_dir():
        tables, stats = landscape_tables(w)
        for name, df in tables.items():
            manifest["tables"][name] = write(name, df, {"worktree": str(w), **stats},
                                             "recomputed from raw landscape evaluation records")
    else:
        manifest["landscape"] = ("raw landscape records not found at %s; "
                                 "landscape tables were left unchanged" % w)
        print(manifest["landscape"])

    (OUT / "SOURCES.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print("wrote data/records/SOURCES.json")


if __name__ == "__main__":
    main()

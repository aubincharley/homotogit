"""Build ``experiments/index.json``: every experiment, configuration and cell.

Reads only.  Every number in the index is copied or computed from a record file
whose location (working tree, or ``<commit>:<path>`` for teammates' branches) is
stored beside it, so the index can be audited line by line and rebuilt:

    py scripts/build_experiment_index.py

What is *derived* rather than copied is labelled as such:

* ``summary_final_path`` -- which evaluation path the summary's
  ``final_test_acc`` equals in the last metrics record (exact float equality);
* ``asset_set`` -- identified by content digests, never by batch name;
* ``code_commit_estimate`` -- the last commit touching the launcher before the
  kernel's launch time (kernel bundles are not versioned, so the exact shipped
  commit is not recoverable from git alone);
* mean / sample SD over **numerically valid** seeds; SD is ``null`` with a stated
  reason when fewer than two valid seeds exist.

Nothing here selects a best epoch: every accuracy is the run's final record.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.method_labels import (ABLATION_LABEL, ADAPTIVE_LABEL,
                                   campaign_label, resbench_label)
from scripts.records import (ROOT, Location, Reader, git, introducing_commit,
                             list_files, resolve)

OUT = ROOT / "experiments" / "index.json"
ASSET_DIR = ROOT / "experiments" / "assets"
SCHEMA_VERSION = 1

#: teammates' branches, pinned to the commits the records were read from
PINNED_REFS = {
    "idriss_ablation": {"ref": "origin/continuation-gaussian-tv_exploration_1",
                        "commit": "591e125"},
    "aubin_sigma0": {"ref": "origin/adaptative-schedule", "commit": "ea186fe"},
}

REFERENCE_RECIPE = {
    "dataset": "CIFAR-10, official split: 50,000 train / 10,000 test",
    "augmentation": "none",
    "architecture": "resnet20_bn_cifar: ResNet-20, BatchNorm (momentum 0.1), "
                    "option-A shortcuts, 269,722 parameters",
    "optimizer": "SGD lr 0.005, momentum 0.9, weight decay 5e-4 (all parameters), "
                 "nesterov off; 60 warmup updates then cosine to 0, indexed by "
                 "global update",
    "batch": "effective 128 via microbatches of 32, loss weighted by example "
             "count; final partial batch 80 = 32+32+16",
    "budget": "30 epochs = 11,730 updates (391 per epoch)",
    "normalization": "per-channel mean/std of the 50,000 unfiltered training "
                     "images, float64 then float32",
}

TIMING_SCOPE = {
    "kaggle_cell": "wall_seconds from model construction to summary write, "
                   "including evaluations and checkpoint writes; eval_seconds "
                   "counts snapshot evaluation only; train_seconds = wall - eval "
                   "(so it includes data indexing and checkpoint I/O). One T4, "
                   "two cells in parallel on a two-GPU kernel.",
    "pilot": "wall_seconds of the run as recorded by continuation_driver; "
             "includes periodic evaluation.",
    "cpu": "CPU study; seconds are not comparable with GPU cells.",
    None: "not recorded",
}

#: experiment metadata that is not written in the records themselves
EXPERIMENTS = [
    dict(id="exp0_input_gaussian", kind="training_pilot", owner="Max",
         kb="EXP-000", title="Fixed Gaussian input blur, GroupNorm",
         groups=[("results/exp0_gaussian", None)], layout="exp0",
         conditions="CIFAR-10 45,000 train / 5,000 stratified val; ResNet-20 "
                    "GroupNorm; 14,040 updates; fixed sigma per run",
         code=["continuation/experiments/exp0.py", "configs/exp0_gaussian.yaml"]),
    dict(id="exp1_input_warmstart", kind="training_pilot", owner="Max",
         kb="EXP-001", title="Input-blur warm starts, GroupNorm",
         groups=[("results/exp1_gaussian_warmstart", None)], layout="exp0",
         conditions="as exp0; warm-start branching from blurred prefixes",
         code=["continuation/experiments/exp1.py", "configs/exp1_warmstart.yaml"]),
    dict(id="tv_budget_previews", kind="preview_no_training", owner="Max",
         kb="EXP-002", title="TV-L2 / TV-H^-1 budget projections (image previews)",
         groups=[("results/tv_previews", None)], layout="artifacts",
         conditions="ten CIFAR-10 images, Chambolle-Pock solves; no classifier "
                    "was trained. Unrelated to the resbench 'l2' / 'hminus1' "
                    "reduction operators, which are constrained quadratic "
                    "down-sampling layers trained inside the network.",
         code=["continuation/transforms/tv.py", "scripts/tv_previews.py"]),
    dict(id="wavelet_previews", kind="preview_no_training", owner="Max",
         kb="EXP-003", title="Undecimated wavelet shrinkage previews and cost",
         groups=[("results/wavelet_previews", None)], layout="artifacts",
         conditions="image previews and micro-benchmarks; no classifier trained",
         code=["continuation/transforms/wavelet.py", "scripts/wavelet_previews.py"]),
    dict(id="pilot_internal_gaussian", kind="training_pilot", owner="Max",
         kb="EXP-004", title="First internal-filter pilots (prelim checks, one "
                             "collapsed run)",
         groups=[("results/kaggle_outputs/pilot-continuation-20260908-102201", None),
                 ("results/kaggle_outputs/pilot-continuation-20260908-102533", None),
                 ("results/kaggle_outputs/pilot-continuation-20260908-103434", None),
                 ("results/kaggle_outputs/pilot-continuation-20260908-122558", None),
                 ("results/kaggle_outputs/pilot-manual", None)],
         layout="pilot_manual", timing="pilot",
         conditions="10,000-image subsets, ResNet-20 GroupNorm; the first pilot "
                    "collapsed to uniform prediction (LR too high)",
         code=["scripts/kaggle_pilot_continuation.py"]),
    dict(id="gn_lr_audit", kind="training_pilot", owner="Max", kb="EXP-005",
         title="GroupNorm LR diagnosis and 10,000-image plain vs Gaussian",
         groups=[("results/kaggle_outputs/lr-diagnostic-20260908-132240", None),
                 ("results/kaggle_outputs/lr-control-002-20260908-140604", None),
                 ("results/kaggle_outputs/plain-study-20260908-132611", None),
                 ("results/kaggle_outputs/gaussian-study-20260908-132957", None)],
         layout="pilot", timing="pilot",
         conditions="10,000 train images, 5,000 val; ResNet-20 GroupNorm; 1,200 "
                    "updates; val accuracy, not test",
         code=["scripts/continuation_driver.py", "scripts/_study_common.py"]),
    dict(id="resnet18bn_gaussian_pilot", kind="training_pilot", owner="Max",
         kb="EXP-006", title="ResNet-18 BN, plain vs internal Gaussian",
         groups=[("results/kaggle_outputs/resnet18-gaussian-20260908-144640", None),
                 ("results/kaggle_outputs/resnet18-gaussian-complete", None)],
         layout="pilot", timing="pilot",
         conditions="10,000-image subset, val split; ResNet-18 BatchNorm; seed 0",
         code=["scripts/job_resnet18_gaussian.py", "scripts/continuation_driver.py"]),
    dict(id="resnet20bn_gaussian_pilot", kind="training_pilot", owner="Max",
         kb="EXP-007", title="ResNet-20 BN pilot, plain vs internal Gaussian",
         groups=[("results/kaggle_outputs/resnet20bn-gaussian-20260908-154226", None)],
         layout="pilot", timing="pilot",
         conditions="10,000-image subset, 2,400 updates, val split; seed 0",
         code=["scripts/job_resnet20bn_gaussian.py", "scripts/continuation_driver.py"]),
    dict(id="fulldata_gaussian", kind="training_benchmark", owner="Max",
         kb="EXP-008", title="Full-data plain / Gaussian plateau / geometric",
         groups=[("results/kaggle_outputs/fulldata-r20bn-20260908-161221", None)],
         layout="pilot", timing="pilot", recipe=True,
         conditions="reference recipe; evaluation every 2 epochs; this run "
                    "produced the pinned campaign asset set",
         code=["scripts/job_fulldata_campaign.py", "scripts/continuation_driver.py"]),
    dict(id="db2_pilot", kind="training_pilot", owner="Max", kb="EXP-009",
         title="db2 wavelet shrinkage pilot (thread closed)",
         groups=[("results/kaggle_outputs/db2-pilot-r20bn-20260908-191741", None)],
         layout="pilot", timing="pilot",
         conditions="10,000-image subset, 2,400 updates, val split; seed 0",
         code=["scripts/job_db2_pilot.py", "scripts/continuation_driver.py"]),
    dict(id="progressive_resolution_pilot", kind="training_pilot", owner="Max",
         kb="EXP-010", title="Progressive input resolution, one seed",
         groups=[("results/kaggle_outputs/progres-r20bn-20260909-075846", None)],
         layout="pilot", timing="pilot", recipe=True,
         conditions="reference recipe, seed 0; pairs with fulldata_gaussian seed 0",
         code=["scripts/job_progressive_resolution.py", "scripts/continuation_driver.py"]),
    dict(id="campaign_grid21", kind="training_benchmark", owner="Max",
         kb="EXP-011", title="21-configuration grid: resolution x blur x site",
         groups=[("results/kaggle_outputs/campaign-j0-20260909-095039", None),
                 ("results/kaggle_outputs/campaign-j1-20260909-095101", None),
                 ("results/kaggle_outputs/campaign-j2-20260909-095123", None)],
         layout="kaggle", timing="kaggle_cell", recipe=True,
         conditions="reference recipe; evaluation every 2 epochs",
         code=["scripts/job_campaign_0.py", "scripts/campaign_driver.py",
               "scripts/campaign_manifest.py", "continuation/campaign_ops.py"]),
    dict(id="ablation_aa", kind="training_benchmark", owner="Idriss",
         kb="EXP-012 (branch continuation-gaussian-tv_exploration_1)",
         title="Anti-aliasing ablation: placement, masks, constant sigma, "
               "BlurPool, internal reductions, per-layer profiles",
         groups=[("results/kaggle_outputs/abl-j%d-%s" % (j, t), "idriss_ablation")
                 for t, js in (("20260910-091356", 0), ("20260910-091418", 1),
                               ("20260910-091443", 2), ("20260910-091504", 3))
                 for j in [js]]
                + [("results/kaggle_outputs/%s" % d, "idriss_ablation") for d in (
                    "abl2-j0-20260910-105903", "abl2-j1-20260910-105925",
                    "abl2-j2-20260910-105951", "abl2-j3-20260910-110013",
                    "abl3-j0-20260910-120721", "abl3-j1-20260910-120744",
                    "abl3-j2-20260910-120812", "abl3-j3-20260910-120834",
                    "abl4-j0-20260910-124807", "abl4-j1-20260910-124830",
                    "abl4-j2-20260910-124857", "abl4-j3-20260910-124919",
                    "abl5-j0-20260910-135809", "abl5-j1-20260910-135831",
                    "abl5-j2-20260910-135858", "abl5-j3-20260910-135920")],
         layout="kaggle", timing="kaggle_cell", recipe=True,
         conditions="reference recipe; own pinned initial weights "
                    "(r20bn-ablation-assets); evaluation every 2 epochs",
         code=["scripts/ablation_manifest.py", "scripts/ablation2_manifest.py",
               "scripts/ablation3_manifest.py", "continuation/ablation_ops.py"]),
    dict(id="adaptive_continuation", kind="training_benchmark", owner="Aubin",
         kb="docs/experiments_2026-09-10.md, docs/adaptive_continuation_maths.md",
         title="Data-triggered schedules, gradient-norm calibration, dwell "
               "allocation, long-horizon runs",
         groups=[("results/kaggle_outputs/%s" % d, None) for d in (
             "adaptive-sigma-20260910-093625", "adaptive-sigma-20260910-093945",
             "adaptive-gap-20260910-115703", "adaptive-gap2-20260910-121319",
             "adaptive-res-20260910-131553", "gradnorm-cal-20260910-105845",
             "gradnorm-cal2-20260910-111547", "signals-cal3-20260910-113759",
             "alloc-search-20260910-132825", "alloc-search-20260910-141638",
             "long-conv-20260910-123736", "long-conv-20260910-124115")],
         layout="kaggle", timing="kaggle_cell", recipe=True,
         conditions="reference recipe, mostly seed 0 only; long-conv runs use "
                    "120 epochs and are not comparable on accuracy",
         code=["scripts/job_adaptive_gap.py", "scripts/job_adaptive_sigma.py",
               "scripts/job_adaptive_resolution.py",
               "scripts/job_gradnorm_calibration.py",
               "scripts/job_allocation_search.py", "scripts/job_long_convergence.py"]),
    dict(id="per_layer_sigma", kind="training_benchmark", owner="Aubin",
         kb="EXP-012 (branch continuation-gaussian-tv); EXP-013 (branch "
            "adaptative-schedule)",
         title="Per-layer sigma profiles (rho), adaptive controller, sigma0 sweep",
         groups=[("results/per_layer_cpu", None),
                 ("results/kaggle_outputs/pls-smoke-20260910-093941", None),
                 ("results/kaggle_outputs/pls-smoke-20260910-094522", None),
                 ("results/kaggle_outputs/per-layer-sigma-20260910-094757", None),
                 ("results/kaggle_outputs/per-layer-adaptive-fix-20260910-130520", None),
                 ("results/kaggle_outputs/sigma0-sweep-20260910-203015", "aubin_sigma0")],
         layout="per_layer", timing="kaggle_cell",
         conditions="own self-paired assets per seed (different digest scheme); "
                    "per_layer_cpu is a 6,000/2,000-image CPU pilot of 12 epochs",
         code=["scripts/study_per_layer_cpu.py", "scripts/job_per_layer_gpu.py"]),
    dict(id="resbench_resolution_only", kind="training_benchmark", owner="Max",
         kb=None, title="Resolution-only benchmark: 7 operators x 5 sites x "
                        "4 schedules, all Gaussian disabled",
         groups=[("results/kaggle_outputs/resbench-j0-20260910-214422", None),
                 ("results/kaggle_outputs/resbench-j1-20260910-214450", None),
                 ("results/kaggle_outputs/resbench-j2-20260910-214517", None)],
         layout="kaggle", timing="kaggle_cell", recipe=True,
         conditions="reference recipe; evaluation every 2 epochs; fixed16 / "
                    "fixed24 controls report their reduced-resolution path",
         code=["scripts/job_resbench_0.py", "scripts/resbench_driver.py",
               "scripts/resbench_manifest.py", "continuation/resolution_ops.py"]),
    dict(id="unified_selected", kind="training_benchmark", owner="Max", kb=None,
         title="Promising methods in one batch, evaluated every epoch",
         groups=[("results/kaggle_outputs/unified-j0-20260911-113738", None),
                 ("results/kaggle_outputs/unified-j1-20260911-113756", None),
                 ("results/kaggle_outputs/unified-j2-20260911-114801", None),
                 ("results/kaggle_outputs/unified-j3-20260911-114953", None)],
         layout="kaggle", timing="kaggle_cell", recipe=True,
         conditions="reference recipe; evaluation after every epoch; per-epoch "
                    "mean training loss recorded",
         code=["scripts/job_unified_0.py", "scripts/unified_driver.py",
               "scripts/unified_manifest.py", "continuation/ablation_ops.py"]),
    dict(id="loss_landscape_blur", kind="analysis_no_training", owner="Aubin",
         kb=None, title="Loss-landscape slices around a trained Gaussian checkpoint",
         groups=[("results/landscape", None)], layout="artifacts",
         conditions="2-D filter-normalised slices, filtered vs bypassed objective, "
                    "optional BN recalibration; no training",
         code=["scripts/loss_landscape_blur.py", "scripts/plot_loss_landscape.py"]),
    dict(id="infrastructure_probes", kind="infrastructure_probe", owner="Max",
         kb=None, title="Kaggle accelerator and mount probes",
         groups=[("results/kaggle_outputs/%s" % d, None) for d in (
             "probe-env-20260908-103026", "probe-invalid-20260908-112736",
             "probe-mount-20260908-103235", "probe-t4-20260908-110032",
             "probe-t4-verified-20260908-122444", "probe-maxlefrr-20260909-092522",
             "probe-maxnikezz-20260909-093139", "probe-mmn-20260911-113012")],
         layout="artifacts", conditions="environment probes only",
         code=["scripts/kaggle_probe_env.py"]),
]

#: arms whose final inference path is not the plain 32x32 ResNet-20
CONTROL_FLAGS = {
    ("resbench_resolution_only", r"__fixed16$"):
        "fixed-resolution control: trained and evaluated at 16x16 after block/"
        "site reduction; final metric is that reduced path, the 32x32 path is a "
        "diagnostic",
    ("resbench_resolution_only", r"__fixed24$"):
        "fixed-resolution control: trained and evaluated at 24x24; final metric "
        "is that reduced path",
    ("ablation_aa", r"^K_const"):
        "fixed-filter control: constant sigma, never annealed; final metric is "
        "the filtered (current) path",
    ("ablation_aa", r"^R6$"):
        "fixed-filter control: constant post-ReLU sigma; final metric is the "
        "filtered (current) path",
    ("ablation_aa", r"^B_blurpool"):
        "fixed BlurPool architecture kept at inference: its target path is not "
        "the plain ResNet-20",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def mean_sd(values):
    if not values:
        return None, None, "no valid seeds"
    if len(values) == 1:
        return values[0], None, "unavailable: one valid seed"
    return st.mean(values), st.stdev(values), None


PATH_KEYS = {
    "current": ("_current", "_filtered"),
    "target": ("_target", "_bypass32", ""),
}


def path_values(rec: dict, split: str, metric: str) -> dict:
    """``{"current": v, "target": v}`` from one metrics record."""
    out = {}
    for path, suffixes in PATH_KEYS.items():
        for s in suffixes:
            k = "%s_%s%s" % (split, metric, s)
            if k in rec and rec[k] is not None:
                out[path] = rec[k]
                break
    return out


def last_record(metrics):
    if isinstance(metrics, dict):
        for k in ("records", "history", "metrics"):
            if isinstance(metrics.get(k), list):
                metrics = metrics[k]
                break
    return metrics[-1] if isinstance(metrics, list) and metrics else None


def all_ce_finite(metrics) -> bool | None:
    if isinstance(metrics, dict):
        return None
    seen = False
    for rec in metrics or []:
        for k, v in rec.items():
            if re.search(r"(^|_)ce(_|$)", k) and isinstance(v, (int, float)):
                seen = True
                if not math.isfinite(v):
                    return False
    return True if seen else None


def digest_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------
# asset sets
# --------------------------------------------------------------------------

def known_asset_sets(reader: Reader, refs: dict) -> dict:
    """Reference manifests, written to experiments/assets/ from tracked sources."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    sets = {}

    camp = json.loads((ROOT / "results/campaign_manifest_frozen.json")
                      .read_text(encoding="utf-8"))["assets"]
    sets["r20bn-campaign-assets"] = {
        "source": "results/campaign_manifest_frozen.json#assets",
        "produced_by": "fulldata_gaussian (%s / %s)" % (camp["source_job"],
                                                         camp["source_study"]),
        "staged_by": "scripts/stage_campaign_assets.py, one private Kaggle "
                     "dataset per account (<account>/r20bn-campaign-assets)",
        "arrays": {k: v["sha256"] for k, v in camp["arrays"].items()},
        "states": dict(camp["states"]),
        "files": {k: v["sha256"] for k, v in camp["files"].items()},
    }

    loc = Location("results/kaggle_outputs/abl-j0-20260910-091356/"
                   "ablation_job0_20260910-091412/assets_verification.json",
                   refs["idriss_ablation"])
    man = reader.read_json(loc)["manifest"]
    sets["r20bn-ablation-assets"] = {
        "source": str(loc),
        "produced_by": man.get("generated_by"),
        "staged_by": "idrisselkhamlichi/r20bn-ablation-assets",
        "arrays": {k: v["sha256"] for k, v in man["arrays"].items()},
        "states": {k: (v["sha256"] if isinstance(v, dict) else v)
                   for k, v in man["states"].items()},
        "files": {k: v["sha256"] for k, v in man.get("files", {}).items()},
        "note": man.get("not_paired_with"),
    }
    for name, s in sets.items():
        s["content_id"] = digest_of({"arrays": s["arrays"], "states": s["states"]})[:16]
        (ASSET_DIR / (name + ".json")).write_text(json.dumps(s, indent=2) + "\n",
                                                  encoding="utf-8")
    return sets


def compare_to_sets(arrays: dict, states: dict, sets: dict) -> dict:
    """Which known set these digests match, component by component."""
    best = None
    for name, s in sets.items():
        a_keys = [k for k in arrays if k in s["arrays"]]
        s_keys = [k for k in states if k in s["states"]]
        if not a_keys and not s_keys:
            continue
        a_ok = all(arrays[k] == s["arrays"][k] for k in a_keys)
        s_ok = all(states[k] == s["states"][k] for k in s_keys)
        res = {"set": name, "arrays_compared": a_keys, "arrays_match": a_ok,
               "states_compared": s_keys, "states_match": s_ok}
        if a_ok and s_ok:
            return res
        if best is None or (a_ok and not best["arrays_match"]):
            best = res
    return best or {"set": None}


def group_assets(reader, loc_base: Location, files: list, sets: dict,
                 launch: dict | None) -> dict:
    """Asset evidence for one output group, strongest first."""
    by_name = defaultdict(list)
    for f in files:
        by_name[f.rsplit("/", 1)[-1]].append(f)

    if loc_base.path.endswith("fulldata-r20bn-20260908-161221"):
        s = sets["r20bn-campaign-assets"]
        return {"evidence": "this run produced the files later pinned as "
                            "r20bn-campaign-assets (source_job in "
                            "results/campaign_manifest_frozen.json)",
                "arrays": s["arrays"], "states": s["states"],
                "match": {"set": "r20bn-campaign-assets", "arrays_match": True,
                          "states_match": True, "by": "producer"}}

    for f in by_name.get("assets_manifest.json", []):
        m = reader.read_json(Location(f, loc_base.ref))
        arrays = {k: v["sha256"] for k, v in m.get("arrays", {}).items()}
        states = dict(m.get("states", {}))
        cmp = compare_to_sets(arrays, states, sets)
        return {"evidence": "digests in %s" % f, "arrays": arrays, "states": states,
                "match": cmp, "pairing_checks": m.get("pairing_checks"),
                "declared_paired_with_campaign": m.get("paired_with_campaign"),
                "declared_source": m.get("source_study") or m.get("source_job"),
                "note": m.get("note")}

    for f in by_name.get("assets_verification.json", []):
        v = reader.read_json(Location(f, loc_base.ref))
        if "manifest" in v:
            m = v["manifest"]
            arrays = {k: x["sha256"] for k, x in m.get("arrays", {}).items()}
            states = {k: (x["sha256"] if isinstance(x, dict) else x)
                      for k, x in m.get("states", {}).items()}
            return {"evidence": "manifest embedded in %s" % f, "arrays": arrays,
                    "states": states, "match": compare_to_sets(arrays, states, sets),
                    "declared_paired_with_campaign": m.get("paired_with_campaign"),
                    "declared_source": m.get("source_study") or m.get("source_job"),
                    "verified": v.get("all_match")}
        slugs = (launch or {}).get("datasets") or []
        assets_slug = [s for s in slugs if s.endswith("r20bn-campaign-assets")]
        if assets_slug and v.get("all_match"):
            s = sets["r20bn-campaign-assets"]
            return {"evidence": "boolean sha256 verification (%s) against the "
                                "manifest shipped in Kaggle dataset %s; that "
                                "dataset was staged from files whose digests are "
                                "recorded in results/campaign_manifest_frozen.json"
                                % (f, assets_slug[0]),
                    "arrays": s["arrays"], "states": s["states"],
                    "checks": v.get("checks"),
                    "match": {"set": "r20bn-campaign-assets", "arrays_match": True,
                              "states_match": True, "by": "dataset slug + boolean "
                                                          "verification"}}

    for f in by_name.get("pairing_verification.json", []):
        v = reader.read_json(Location(f, loc_base.ref))
        if "observed" in v:
            obs = v["observed"]
            arrays = {k: x for k, x in obs.items() if not k.startswith("init")}
            states = {k: x for k, x in obs.items() if k.startswith("init")}
            return {"evidence": "observed digests in %s" % f, "arrays": arrays,
                    "states": states, "match": compare_to_sets(arrays, states, sets)}
        if "per_seed" in v:
            return {"evidence": "self-paired digests in %s (different digest "
                                "scheme: combined 'perms' and 'init')" % f,
                    "per_seed": {s: x.get("expected") for s, x in v["per_seed"].items()},
                    "match": {"set": "per-layer-self-paired",
                              "arrays_match": None, "states_match": None,
                              "note": "subset digest equals the campaign subset; "
                                      "init and permutation digests use another "
                                      "scheme and cannot be compared"},
                    "scope": v.get("pairing_scope")}
    return {"evidence": None, "match": {"set": None}}


# --------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------

def cell_identity(exp_id: str, layout: str, summary: dict, path: str):
    """``(config_id, seed)`` for one summary."""
    seed = summary.get("seed")
    if layout == "kaggle":
        cid = summary.get("id")
        if not cid and summary.get("cell_id"):
            cid = str(summary["cell_id"]).rsplit("__seed", 1)[0]
        return cid, seed
    if layout == "per_layer":
        arm = summary.get("arm") or path.split("/")[-2]
        arm = re.sub(r"_seed\d+$", "", arm)
        if summary.get("sigma0") is not None:
            arm = "%s@sigma0=%g" % (arm, summary["sigma0"])
        return arm, seed
    if layout == "pilot_manual":
        m = re.search(r"seed(\d+)", path)
        return summary.get("variant") or path.split("/")[-2], (
            int(m.group(1)) if m else seed)
    if layout == "exp0":
        name = summary.get("run_name") or path.split("/")[-2]
        m = re.match(r"(.+)__seed_(\d+)$", name)
        return (m.group(1), int(m.group(2))) if m else (name, seed)
    label = summary.get("label") or path.split("/")[-2]
    return re.sub(r"_seed\d+$", "", str(label)), seed


def label_for(exp_id: str, cid: str, summary: dict) -> tuple[str, str | None]:
    if exp_id == "campaign_grid21":
        return campaign_label(cid)
    if exp_id == "resbench_resolution_only":
        return resbench_label(cid)
    if exp_id == "ablation_aa":
        return ABLATION_LABEL.get(cid, (cid, None))
    if exp_id == "unified_selected":
        return summary.get("label", cid), None
    if exp_id in ("adaptive_continuation", "per_layer_sigma"):
        base = cid.split("@")[0]
        lab, fam = ADAPTIVE_LABEL.get(base, (base.replace("_", " "), None))
        if base.startswith("s0_"):
            lab = ADAPTIVE_LABEL["rho1"][0]
        if "@sigma0=" in cid:
            lab += ", sigma0 = %s" % cid.split("=")[1]
        return lab, fam
    return str(summary.get("label") or summary.get("variant") or cid), None


def extract_cell(reader, exp, loc: Location, files_in_dir: set) -> dict:
    s = reader.read_json(loc)
    d = loc.path.rsplit("/", 1)[0]
    cid, seed = cell_identity(exp["id"], exp["layout"], s, loc.path)
    metrics_loc = None
    for name in ("metrics.json", "metrics.jsonl"):
        if d + "/" + name in files_in_dir:
            metrics_loc = Location(d + "/" + name, loc.ref)
            break
    metrics = None
    if metrics_loc is not None and metrics_loc.path.endswith(".json"):
        try:
            metrics = reader.read_json(metrics_loc)
        except Exception:
            metrics = None
    last = last_record(metrics)

    split = ("test" if any(k.startswith("final_test") for k in s) else
             "val" if any(k.startswith("final_val") for k in s) else None)
    final = s.get("final_test_acc", s.get("final_val_acc"))
    final_ce = s.get("final_test_ce", s.get("final_val_ce"))
    if exp["layout"] == "exp0" and isinstance(s.get("final"), dict):
        # input-space runs: "target_val" is the unfiltered image, the endpoint
        f = s["final"]
        final = (f.get("target_val") or {}).get("accuracy")
        final_ce = (f.get("target_val") or {}).get("ce")
        split = "val"
    if exp["layout"] == "pilot_manual" and "full_val_acc_unfiltered" in s:
        final, final_ce, split = (s["full_val_acc_unfiltered"],
                                  s["full_val_ce_unfiltered"], "val")

    acc_paths = path_values(last, split, "acc") if (last and split) else {}
    ce_paths = path_values(last, split, "ce") if (last and split) else {}
    final_path = None
    if exp["layout"] == "exp0" and isinstance(s.get("final"), dict):
        final_path = "target (unfiltered val images)"
    elif exp["layout"] == "pilot_manual" and "full_val_acc_unfiltered" in s:
        final_path = "target (unfiltered)"
    if acc_paths and final is not None:
        hits = [p for p, v in acc_paths.items() if v == final]
        final_path = ("current=target" if len(hits) == 2 else
                      hits[0] if hits else "unmatched")
    if s.get("primary_path"):
        declared = s["primary_path"]
        final_path = final_path or declared
    if exp["id"] == "ablation_aa" and s.get("final_test_acc_primary") is not None:
        final = s["final_test_acc_primary"]
        final_ce = (s.get("final_test_ce_current") if s.get("primary_path") == "current"
                    else s.get("final_test_ce"))
        final_path = "%s (declared primary_path)" % s.get("primary_path")

    ce_ok = all_ce_finite(metrics)
    valid = finite(final_ce) if final_ce is not None else (ce_ok is not False)
    if ce_ok is False:
        valid = False

    state_end = (last or {}).get("current") if isinstance((last or {}).get("current"), dict) else None
    return {
        "cell_id": s.get("cell_id") or "%s__seed%s" % (cid, seed),
        "config_id": cid, "seed": seed,
        "status": "completed" if not s.get("incomplete") else "incomplete",
        "numerically_valid": bool(valid),
        "all_recorded_ce_finite": ce_ok,
        "summary": loc.as_dict(), "metrics": metrics_loc.as_dict() if metrics_loc else None,
        "n_records": len(metrics) if isinstance(metrics, list) else None,
        "split": split, "n_train": s.get("n_train"),
        "n_eval": s.get("n_test", s.get("n_val")),
        "epochs": s.get("epochs"), "updates": s.get("updates", s.get("updates_completed")),
        "final_acc": final, "final_ce": final_ce,
        "summary_final_path": final_path,
        "last_record_acc_by_path": acc_paths or None,
        "last_record_ce_by_path": ce_paths or None,
        "current_equals_target_at_end": (acc_paths.get("current") == acc_paths.get("target")
                                         if len(acc_paths) == 2 else None),
        "state_at_end": state_end,
        "final_train_probe_ce": s.get("final_train_probe_ce",
                                      s.get("final_train_probe_ce_bypassed")),
        "wall_seconds": s.get("wall_seconds"), "train_seconds": s.get("train_seconds"),
        "eval_seconds": s.get("eval_seconds"), "peak_mem_mib": s.get("peak_mem_mib"),
        "device": s.get("gpu") or s.get("device"),
        "controller": {k: s.get("controller", {}).get(k) for k in (
            "operator", "n_sites", "placement", "reduction", "reduction_point",
            "insertion_mask", "resolution_by_epoch", "levels", "sigma_profile",
            "location", "path") if isinstance(s.get("controller"), dict)
            and k in s["controller"]} or None,
        "_summary": s,
    }


def exp0_like_cells(reader, exp, prefix: str):
    files = list_files(prefix)
    out = []
    fset = set(files)
    for f in files:
        if f.endswith("/summary.json") and f.count("/") >= 3:
            out.append(extract_cell(reader, exp, Location(f), fset))
    return out


def attempted_cells(reader, base_ref, files) -> tuple[set, list]:
    attempted, failed = set(), []
    for f in files:
        name = f.rsplit("/", 1)[-1]
        if name == "job_manifest.json":
            m = reader.read_json(Location(f, base_ref))
            for c in m.get("cells", []):
                attempted.add(c["cell_id"] if isinstance(c, dict) else str(c))
        elif name in ("FAILED.json", "ABORTED.json"):
            d = f.rsplit("/", 2)
            failed.append({"marker": name, "location": {"ref": base_ref, "path": f},
                           "cell_id": d[-2] if len(d) >= 2 else None})
    return attempted, failed


def launch_info(reader, base_ref, prefix, files):
    lf = prefix + "/launch.json"
    if lf in files:
        return reader.read_json(Location(lf, base_ref))
    return None


#: The kaggle launcher ships a copy of the code in ``<group>/_repo`` (git-ignored,
#: present only on the machine that downloaded the outputs).  When it exists it
#: is the code that actually ran; compare it with candidate commits.
SHIPPED_CHECK = {
    "unified_selected": ("3829fa3", ["continuation/ablation_ops.py",
                                     "continuation/campaign_ops.py",
                                     "continuation/transforms/gaussian.py",
                                     "continuation/models/resnet20_bn.py",
                                     "continuation/pipeline.py", "continuation/optim.py",
                                     "continuation/data.py", "scripts/unified_driver.py",
                                     "scripts/unified_manifest.py",
                                     "scripts/ablation_manifest.py",
                                     "scripts/ablation2_manifest.py"]),
    "resbench_resolution_only": ("61b58e7", ["continuation/resolution_ops.py",
                                             "continuation/campaign_ops.py",
                                             "continuation/transforms/gaussian.py",
                                             "continuation/models/resnet20_bn.py",
                                             "continuation/pipeline.py",
                                             "scripts/resbench_driver.py",
                                             "scripts/resbench_manifest.py"]),
}


def shipped_code_check(exp_id: str, prefix: str, ref) -> dict | None:
    if exp_id not in SHIPPED_CHECK or ref is not None:
        return None
    import os
    bundle_root = Path(os.environ.get("BUNDLE_ROOT", ROOT))
    repo = bundle_root / prefix / "_repo"
    if not repo.is_dir():
        return {"status": "bundle not present under %s (git-ignored; set BUNDLE_ROOT to "
                          "the checkout that downloaded the outputs)" % bundle_root.name}
    commit, files = SHIPPED_CHECK[exp_id]
    rows = {}
    for f in files:
        if not (repo / f).is_file():
            rows[f] = "absent from bundle"
            continue
        shipped = (repo / f).read_bytes()
        blob = git("show", "%s:%s" % (commit, f)).encode("utf-8")
        rows[f] = ("identical" if shipped == blob else
                   "identical after CRLF->LF" if shipped.replace(b"\r\n", b"\n") == blob
                   else "DIFFERENT")
    return {"compared_with_commit": resolve(commit), "files": rows,
            "status": ("matches" if all(v.startswith("identical") for v in rows.values())
                       else "differs"),
            "bundle": str(repo.relative_to(bundle_root)).replace("\\", "/"),
            "note": "Windows checkout with core.autocrlf=true ships CRLF line endings; "
                    "content is compared after normalisation"}


def code_commit_estimate(launch: dict | None, ref_commit: str | None) -> dict | None:
    if not launch or not launch.get("script") or not launch.get("launched_utc"):
        return None
    script = launch["script"].replace("\\", "/")
    rev = ref_commit or "HEAD"
    out = git("log", rev, "-1", "--format=%H", "--before=%s" % launch["launched_utc"],
              "--", script).strip()
    return {"commit": out or None, "script": script,
            "empty_because": (None if out else "no commit touched the launcher before "
                              "launch; it was committed afterwards (see shipped_code_check)"),
            "method": "last commit touching the launcher before launched_utc on "
                      "the branch holding the records; the shipped bundle is not "
                      "versioned, so uncommitted edits at launch time cannot be ruled out"}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    refs = {k: resolve(v["commit"]) for k, v in PINNED_REFS.items()}
    reader = Reader()
    sets = known_asset_sets(reader, refs)

    experiments, configurations, cells_out = [], [], []
    for exp in EXPERIMENTS:
        exp_cells, groups_out = [], []
        for prefix, ref_key in exp["groups"]:
            ref = refs.get(ref_key) if ref_key else None
            files = list_files(prefix, ref)
            if not files:
                groups_out.append({"location": {"ref": ref, "path": prefix},
                                   "status": "missing"})
                continue
            fset = set(files)
            launch = launch_info(reader, ref, prefix, fset)
            g = {"location": {"ref": ref, "path": prefix,
                              "ref_name": PINNED_REFS[ref_key]["ref"] if ref_key else None},
                 "n_files": len(files),
                 "introduced_by_commit": introducing_commit(prefix, ref or "HEAD"),
                 "kaggle_kernel": ({"account": launch.get("account"),
                                    "slug": launch.get("slug"),
                                    "launched_utc": launch.get("launched_utc"),
                                    "datasets": launch.get("datasets")}
                                   if launch else None),
                 "code_commit_estimate": code_commit_estimate(launch, ref),
                 "shipped_code_check": shipped_code_check(exp["id"], prefix, ref),
                 "checkpoints": "not versioned (*.pt is git-ignored); present only "
                                "in the Kaggle kernel output and on the machine "
                                "that downloaded it"}
            if exp["layout"] in ("artifacts",):
                g["artifacts"] = files
                groups_out.append(g)
                continue
            if exp["layout"] == "exp0":
                found = exp0_like_cells(reader, exp, prefix)
            else:
                found = [extract_cell(reader, exp, Location(f, ref), fset)
                         for f in files
                         if f.endswith("/summary.json") and f.count("/") > prefix.count("/") + 1]
            # Aubin's normaliser re-emits a study in the campaign layout next to
            # the original: same run, same numbers, two paths.  Keep one cell
            # per (arm, seed, final numbers) inside a group and record the copy.
            unique = {}
            for c in found:
                key = (c["config_id"], c["seed"], c["final_acc"], c["final_ce"],
                       c["epochs"], c["n_train"])
                if key in unique:
                    keep = unique[key]
                    if "_seed" in c["summary"]["path"].rsplit("/", 2)[-2]:
                        c["duplicate_locations"] = keep.get("duplicate_locations", []) + [keep["summary"]]
                        unique[key] = c
                    else:
                        keep.setdefault("duplicate_locations", []).append(c["summary"])
                else:
                    unique[key] = c
            g["duplicate_summaries_collapsed"] = len(found) - len(unique)
            found = list(unique.values())
            attempted, failed = attempted_cells(reader, ref, files)
            present = {c["cell_id"] for c in found}
            g["assets"] = group_assets(reader, Location(prefix, ref), files, sets, launch)
            g["cells_attempted"] = len(attempted) if attempted else None
            g["cells_with_summary"] = len(found)
            g["failure_markers"] = failed
            g["missing_from_manifest"] = sorted(attempted - present) if attempted else []
            for c in found:
                c["group"] = prefix
                c["asset_set"] = g["assets"]["match"].get("set")
            exp_cells.extend(found)
            groups_out.append(g)

        # configurations
        # a configuration is an arm *under one set of run conditions*: the
        # per-layer study reuses arm names across a CPU pilot and GPU runs
        by_cond = defaultdict(list)
        for c in exp_cells:
            by_cond[(c["config_id"], c["epochs"], c["n_train"], c["n_eval"])].append(c)
        # an arm re-run in another launch with the same seeds is a separate
        # replicate, not extra seeds: keep launches apart when seeds repeat
        by_cfg = defaultdict(list)
        for key, cs in by_cond.items():
            seeds = [c["seed"] for c in cs]
            rerun = len(seeds) != len(set(seeds))
            for c in cs:
                by_cfg[key + ((c["group"] if rerun else None),)].append(c)
        exp_cfgs = []
        for (cid, _ep, _ntr, _nev, _grp), cs in sorted(by_cfg.items(),
                                                       key=lambda kv: tuple(map(str, kv[0]))):
            s0 = cs[0]["_summary"]
            label, family = label_for(exp["id"], cid, s0)
            valid = [c for c in cs if c["numerically_valid"] and c["final_acc"] is not None]
            accs = [c["final_acc"] for c in valid]
            m, sd, why = mean_sd(accs)
            ces = [c["final_ce"] for c in valid if finite(c["final_ce"])]
            cm, csd, _ = mean_sd(ces)
            walls = [c["wall_seconds"] for c in valid if c["wall_seconds"]]
            epochs = sorted({c["epochs"] for c in cs if c["epochs"] is not None})
            flags = [msg for (e, pat), msg in CONTROL_FLAGS.items()
                     if e == exp["id"] and re.search(pat, cid)]
            ends_off_target = [c["cell_id"] for c in cs
                               if c["current_equals_target_at_end"] is False]
            if ends_off_target and not flags:
                flags.append("current path differs from target path at the final "
                             "record for %d cell(s)" % len(ends_off_target))
            if epochs and max(epochs) > 30:
                flags.append("budget %d epochs: not comparable on accuracy with "
                             "30-epoch arms" % max(epochs))
            paths = sorted({str(c["summary_final_path"]) for c in cs})
            conditions = {"epochs": _ep, "n_train": _ntr, "n_eval": _nev,
                          "groups": sorted({c["group"] for c in cs})}
            off_reference = (exp.get("recipe") and (_ep, _ntr, _nev) != (30, 50000, 10000))
            if off_reference or exp["id"] == "per_layer_sigma" and (_ntr, _nev) != (50000, 10000):
                label = "%s [%s train / %s eval, %s epochs]" % (label, _ntr, _nev, _ep)
            if exp["id"] == "adaptive_continuation" and cid == "ADAPTGAP" and _grp                     and "adaptive-gap2-" in _grp:
                # both launches name the arm ADAPTGAP; the second is the corrected
                # trigger that the earlier tables called ADAPTGAP2
                label = ADAPTIVE_LABEL["ADAPTGAP2"][0]
            if _grp:
                label = "%s [launch %s]" % (label, _grp.rsplit("/", 1)[-1])
            cfg = {
                "experiment": exp["id"], "config_id": cid, "label": label,
                "run_conditions": conditions,
                "family": family,
                "seeds_attempted": sorted(c["seed"] for c in cs if c["seed"] is not None),
                "seeds_valid": sorted(c["seed"] for c in valid if c["seed"] is not None),
                "seeds_diverged": sorted(c["seed"] for c in cs if not c["numerically_valid"]),
                "status": ("valid" if len(valid) == len(cs) else
                           "diverged" if not valid else "partially_valid"),
                "n_valid": len(valid),
                "acc_mean": m, "acc_sd": sd, "sd_status": why,
                "acc_per_seed": {str(c["seed"]): c["final_acc"] for c in cs},
                "ce_mean": cm, "ce_sd": csd,
                "split": cs[0]["split"], "epochs": epochs,
                "final_path": paths[0] if len(paths) == 1 else paths,
                "asset_sets": sorted({str(c.get("asset_set")) for c in cs}),
                "wall_seconds_mean": st.mean(walls) if walls else None,
                "flags": flags,
                "controller": cs[0]["controller"],
            }
            exp_cfgs.append(cfg)
        configurations.extend(exp_cfgs)

        for c in exp_cells:
            c = {k: v for k, v in c.items() if k != "_summary"}
            c["experiment"] = exp["id"]
            cells_out.append(c)

        sets_used = sorted({str(g.get("assets", {}).get("match", {}).get("set"))
                            for g in groups_out if "assets" in g})
        n_cells = len(exp_cells)
        n_valid = sum(1 for c in exp_cells if c["numerically_valid"])
        attempted = sum((g.get("cells_attempted") or 0) for g in groups_out)
        experiments.append({
            "id": exp["id"], "title": exp["title"], "kind": exp["kind"],
            "owner": exp["owner"], "knowledge_base_record": exp["kb"],
            "conditions": exp["conditions"],
            "reference_recipe": REFERENCE_RECIPE if exp.get("recipe") else None,
            "code": exp["code"], "timing_scope": TIMING_SCOPE[exp.get("timing")],
            "groups": groups_out,
            "asset_sets": sets_used,
            "counts": {"cells_attempted_per_manifests": attempted or None,
                       "cells_with_summary": n_cells, "cells_numerically_valid": n_valid,
                       "cells_diverged": n_cells - n_valid,
                       "failure_markers": sum(len(g.get("failure_markers") or [])
                                              for g in groups_out),
                       "configurations": len(exp_cfgs)},
        })

    observations = cross_batch_observations(configurations, cells_out)
    index = {
        "schema_version": SCHEMA_VERSION,
        "built_from_commit": git("rev-parse", "HEAD").strip(),
        "pinned_refs": {k: {"ref": PINNED_REFS[k]["ref"], "commit": refs[k]}
                        for k in refs},
        "asset_sets": sets,
        "reference_recipe": REFERENCE_RECIPE,
        "experiments": experiments,
        "configurations": configurations,
        "cells": cells_out,
        "observations": observations,
    }
    reader.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(index, indent=1, allow_nan=True) + "\n", encoding="utf-8")
    print("experiments %d, configurations %d, cells %d -> %s"
          % (len(experiments), len(configurations), len(cells_out), OUT))
    for e in experiments:
        print("  %-30s %-22s sets=%s %s" % (e["id"], e["kind"], e["asset_sets"], e["counts"]))
    return index


def cross_batch_observations(configs, cells) -> list:
    """Facts that span batches.  Stated as observations, not as explanations."""
    obs = []
    plain_ids = {"campaign_grid21": "R32__Gnone__input_bilinear__all19",
                 "resbench_resolution_only": "none__none__none",
                 "unified_selected": "plain", "fulldata_gaussian": "plain"}
    rows = {}
    for c in cells:
        pid = plain_ids.get(c["experiment"])
        if pid and (c["config_id"] == pid or (c["experiment"] == "resbench_resolution_only"
                                              and c["config_id"].startswith("none"))):
            rows.setdefault(c["experiment"], {})[str(c["seed"])] = c["final_acc"]
    if rows:
        spread = {}
        for seed in ("0", "1", "2"):
            vals = [r[seed] for r in rows.values() if seed in r]
            if len(vals) > 1:
                spread[seed] = max(vals) - min(vals)
        obs.append({
            "id": "same-assets-plain-differs",
            "statement": "Plain ResNet-20 arms that verified the same pinned "
                         "assets (initial weights, BN buffers, probe, per-epoch "
                         "permutations) finish at different accuracies in "
                         "different batches. Epoch-0 evaluations are identical; "
                         "the trajectories already differ at the first later "
                         "record. The cause is not established: candidate "
                         "factors include non-deterministic GPU kernels, driver "
                         "code differences between batches, and torch/CUDA "
                         "builds. No universal batch offset follows from this.",
            "plain_final_acc_by_experiment_and_seed": rows,
            "max_minus_min_by_seed": spread,
        })
    obs.append({
        "id": "ablation-assets-share-data-order",
        "statement": "r20bn-ablation-assets and r20bn-campaign-assets have identical "
                     "subset, train-probe and per-epoch permutation digests; only "
                     "the initial weights / BN buffers differ. Arms across the two "
                     "sets share data order but not initialization.",
    })
    obs.append({
        "id": "kb-experiment-id-collision",
        "statement": "EXP-012 names two different records on two branches: "
                     "EXP-012_aa_ablation (Idriss, continuation-gaussian-tv_exploration_1) "
                     "and EXP-012_per_layer_sigma (Aubin, continuation-gaussian-tv). "
                     "The index disambiguates by branch; neither file is renamed.",
    })
    return obs


if __name__ == "__main__":
    main()

"""Experiment 1: Gaussian warm starts and an intermediate continuation stage.

Three arms, all at the same total budget ``B = 14,040`` updates and the same
paired seeds:

===== ================= ==================== =====================
Arm   updates 1-1,500   updates 1,501-3,000  updates 3,001-14,040
===== ================= ==================== =====================
A     sigma = 0         sigma = 0            sigma = 0
W     sigma = 1         sigma = 0            sigma = 0
P     sigma = 1         sigma = 0.5          sigma = 0
===== ================= ==================== =====================

``W`` and ``P`` branch from the *same* complete sigma=1 state at update 1,500 for
each seed.  That prefix is computed once and reused computationally, but its
1,500 updates are counted in the budget of both methods.

Arm A reuses the Experiment 0 sigma=0 runs unchanged: same architecture,
normalization, split, preprocessing, optimizer, momentum, weight decay, batch
size, learning-rate schedule, budget and paired initialization.  Experiment 0's
raw results are read-only here.

Stage lengths were fixed before observing any Experiment 1 result.  They are
pragmatic pilot choices, **not** estimates of an optimal switch time from
Experiment 0's validation peaks.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ..config import ExperimentConfig, deep_update, from_dict, to_dict
from ..data import build_dataset, split_fingerprint
from ..diagnostics import summarize_across_seeds
from ..engine import Trainer
from ..metrics import RunLogger, capture_environment, checkpoint_kind, read_metrics
from ..optim import lr_at

ARMS = ("A", "W", "P")


def arm_schedule(arm: str, sweep: dict) -> dict:
    """Piecewise-constant native-parameter schedule for an arm."""
    B = int(sweep["total_steps"])
    warm = int(sweep["warm_steps"])
    mid = int(sweep["mid_steps"])
    warm_sigma = float(sweep["warm_sigma"])
    mid_sigma = float(sweep["mid_sigma"])
    if arm == "A":
        return {"kind": "constant", "params": {"value": 0.0}}
    if arm == "W":
        return {"kind": "piecewise_constant",
                "params": {"values": [warm_sigma, 0.0], "steps_per_stage": [warm, B - warm]}}
    if arm == "P":
        return {"kind": "piecewise_constant",
                "params": {"values": [warm_sigma, mid_sigma, 0.0],
                           "steps_per_stage": [warm, mid, B - warm - mid]}}
    raise KeyError("unknown arm %r; expected one of %s" % (arm, ARMS))


def prefix_dir(out_root, seed: int) -> Path:
    return Path(out_root) / ("prefix_sigma_warm__seed_%d" % seed)


def arm_dir(out_root, arm: str, seed: int) -> Path:
    return Path(out_root) / ("arm_%s__seed_%d" % (arm, seed))


def _run_cfg(base: ExperimentConfig, name: str, seed: int, out_dir: Path,
             schedule: dict) -> ExperimentConfig:
    raw = to_dict(base)
    # The schedule section is *replaced*, not merged: deep-merging an arm's
    # piecewise schedule into the base constant schedule would leave the base's
    # stale ``value`` key behind and break the constructor.  The momentum policy
    # lives in the same section but is a trainer setting, so it is carried over
    # explicitly rather than lost with the rest of the base params.
    base_params = (raw.get("schedule") or {}).get("params") or {}
    params = dict(schedule.get("params") or {})
    params["momentum_at_stage_boundary"] = base_params.get(
        "momentum_at_stage_boundary", "carry")
    raw["schedule"] = {"kind": schedule["kind"], "params": params}
    raw = deep_update(raw, {"run": {"name": name, "seed": int(seed),
                                    "out_dir": str(out_dir)}})
    raw.pop("sweep", None)
    return from_dict(raw)


# --------------------------------------------------------------------------
# Prefix and arms
# --------------------------------------------------------------------------

def run_prefix(cfg: ExperimentConfig, bundle, seed: int, out_root) -> Path:
    """Rerun the sigma=warm prefix and save a complete branchable state.

    Experiment 0 saved only a model+optimizer checkpoint at the *final* update,
    so no state exists at update 1,500 and the prefix must be recomputed.  It is
    run on the **original B-update learning-rate horizon** and merely stopped
    early -- the schedule is not compressed to the prefix length.
    """
    sweep = cfg.sweep
    warm, warm_sigma = int(sweep["warm_steps"]), float(sweep["warm_sigma"])
    d = prefix_dir(out_root, seed)
    ckpt = d / ("checkpoint_step%06d.pt" % warm)
    if ckpt.exists():
        print("[skip] prefix seed %d already computed" % seed, flush=True)
        return ckpt
    run_cfg = _run_cfg(cfg, d.name, seed, d,
                       {"kind": "constant", "params": {"value": warm_sigma}})
    print("\n=== prefix sigma=%g seed=%d: updates 0-%d on the B=%d horizon ==="
          % (warm_sigma, seed, warm, run_cfg.optim.total_steps), flush=True)
    logger = RunLogger(d)
    try:
        trainer = Trainer(run_cfg, bundle, d, logger)
        trainer.fit(max_steps=warm)
    finally:
        logger.close()
    if not ckpt.exists():
        raise RuntimeError("prefix did not produce %s" % ckpt)
    return ckpt


def verify_branch_state(ckpt_path, cfg: ExperimentConfig, arm: str, seed: int) -> dict:
    """Check the branch point before spending compute on the arm."""
    sweep = cfg.sweep
    warm = int(sweep["warm_steps"])
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sched = arm_schedule(arm, sweep)
    from ..schedules import build_schedule
    from ..config import ScheduleConfig
    s = build_schedule(ScheduleConfig(**sched), cfg.optim.total_steps)
    info = {
        "arm": arm, "seed": seed,
        "checkpoint": str(ckpt_path),
        "checkpoint_kind": checkpoint_kind(ckpt_path),
        "global_step": int(ck["step"]),
        "sigma_before_branch": s.parameter(warm - 1),
        "sigma_after_branch": s.parameter(warm),
        "lr_at_branch": lr_at(warm, cfg.optim),
        "lr_horizon_total_steps": cfg.optim.total_steps,
        "optimizer_has_momentum_buffers": any(
            "momentum_buffer" in v for v in ck["optimizer_state"]["state"].values()),
        "batch_stream_epoch": ck["batch_stream"]["epoch"],
        "batch_stream_pos": ck["batch_stream"]["pos"],
        "amp_scaler_state": ck["amp"],
    }
    problems = []
    if info["checkpoint_kind"] != "full_resumable":
        problems.append("checkpoint is not a complete resumable state")
    if info["global_step"] != warm:
        problems.append("checkpoint step %s != warm_steps %s" % (info["global_step"], warm))
    if info["sigma_before_branch"] != float(sweep["warm_sigma"]):
        problems.append("schedule does not start at warm_sigma")
    if not info["optimizer_has_momentum_buffers"]:
        problems.append("no momentum buffers to carry across the branch")
    info["problems"] = problems
    if problems:
        raise RuntimeError("branch verification failed for arm %s seed %d: %s"
                           % (arm, seed, "; ".join(problems)))
    return info


def run_arm(cfg: ExperimentConfig, bundle, arm: str, seed: int, out_root,
            prefix_ckpt=None) -> dict:
    d = arm_dir(out_root, arm, seed)
    if (d / "summary.json").exists():
        print("[skip] arm %s seed %d already complete" % (arm, seed), flush=True)
        with open(d / "summary.json", "r", encoding="utf-8") as fh:
            return json.load(fh)
    run_cfg = _run_cfg(cfg, d.name, seed, d, arm_schedule(arm, cfg.sweep))
    check = None
    if prefix_ckpt is not None:
        check = verify_branch_state(prefix_ckpt, cfg, arm, seed)
    print("\n=== arm %s seed %d (branch from %s) ==="
          % (arm, seed, "scratch" if prefix_ckpt is None else Path(prefix_ckpt).name), flush=True)
    logger = RunLogger(d)
    try:
        if check is not None:
            logger.log({"record": "branch_check", **check})
        trainer = Trainer(run_cfg, bundle, d, logger, resume_from=prefix_ckpt)
        summary = trainer.fit()
    finally:
        logger.close()
    summary["arm"] = arm
    summary["branch_check"] = check
    with open(d / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


def run_experiment(cfg: ExperimentConfig, out_root, seeds=None, arms=None) -> dict:
    seeds = list(seeds if seeds is not None else cfg.sweep["seeds"])
    arms = list(arms if arms is not None else ["W", "P"])
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    bundle = build_dataset(cfg.data)

    manifest = {
        "experiment": "exp1_gaussian_warmstart",
        "arms": {a: arm_schedule(a, cfg.sweep) for a in ARMS},
        "seeds": seeds,
        "sweep": dict(cfg.sweep),
        "base_config": to_dict(cfg),
        "split": split_fingerprint(bundle),
        "baseline_arm_A_source": cfg.sweep.get("baseline_dir"),
        "momentum_at_stage_boundary": "carry",
        "environment": capture_environment(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": [],
    }

    t0 = time.perf_counter()
    for seed in seeds:
        ckpt = run_prefix(cfg, bundle, seed, out_root)
        for arm in arms:
            summary = run_arm(cfg, bundle, arm, seed, out_root, prefix_ckpt=ckpt)
            manifest["runs"].append({
                "arm": arm, "seed": seed, "dir": str(arm_dir(out_root, arm, seed)),
                "parent_checkpoint": str(ckpt),
                "final": summary["final"], "timing": summary["timing"],
            })
            with open(out_root / "manifest.json", "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, default=str)
    manifest["incremental_wall_seconds"] = round(time.perf_counter() - t0, 2)
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(out_root / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return manifest


# --------------------------------------------------------------------------
# Collection and aggregation
# --------------------------------------------------------------------------

def collect_runs(out_root, baseline_dir, seeds) -> list:
    """Arm A comes from Experiment 0 (read-only); W and P from this experiment."""
    runs = []
    for seed in seeds:
        a_dir = Path(baseline_dir) / ("level_0__seed_%d" % seed)
        if (a_dir / "summary.json").exists():
            with open(a_dir / "summary.json", "r", encoding="utf-8") as fh:
                summary = json.load(fh)
            runs.append({"arm": "A", "seed": seed, "dir": str(a_dir), "reused": True,
                         "summary": summary, "metrics": read_metrics(a_dir / "metrics.jsonl")})
        for arm in ("W", "P"):
            d = arm_dir(out_root, arm, seed)
            if (d / "summary.json").exists():
                with open(d / "summary.json", "r", encoding="utf-8") as fh:
                    summary = json.load(fh)
                runs.append({"arm": arm, "seed": seed, "dir": str(d), "reused": False,
                             "summary": summary, "metrics": read_metrics(d / "metrics.jsonl")})
    return runs


def common_checkpoints(runs, eval_every: int, total: int) -> list:
    """Steps present in every arm's *common* grid (dense points excluded)."""
    grids = []
    for r in runs:
        steps = {m["step"] for m in r["metrics"]
                 if m.get("record") == "eval" and m["step"] % eval_every == 0}
        grids.append(steps)
    common = set.intersection(*grids) if grids else set()
    return sorted(s for s in common if s <= total)


def _final(run) -> dict:
    return run["summary"]["final"]


def threshold_step(run, key: str, thresh: float, common: list):
    """First common checkpoint at which target probe CE <= thresh, else None.

    Uses global updates *including* any warm-start pretraining, and never
    interpolates between sparse measurements.
    """
    by_step = {}
    for m in run["metrics"]:
        if m.get("record") == "eval" and m.get("phase") != "post_transition":
            by_step.setdefault(m["step"], m)
    for s in common:
        m = by_step.get(s)
        if m is not None and m[key]["ce"] <= thresh:
            return s
    return None


def aggregate(out_root, baseline_dir, seeds, eval_every=500, total=14040) -> dict:
    runs = collect_runs(out_root, baseline_dir, seeds)
    common = common_checkpoints(runs, eval_every, total)
    by_arm = {a: sorted([r for r in runs if r["arm"] == a], key=lambda r: r["seed"])
              for a in ARMS}

    metrics = ["target_val.accuracy", "target_val.ce",
               "target_train_probe.accuracy", "target_train_probe.ce"]

    def get(run, dotted):
        block, key = dotted.split(".")
        return _final(run)[block][key]

    arms_out = {}
    for arm, rs in by_arm.items():
        if not rs:
            continue
        entry = {"n_seeds": len(rs), "seeds": [r["seed"] for r in rs],
                 "reused_from_exp0": all(r.get("reused") for r in rs),
                 "final": {}, "thresholds": {}, "timing": {}}
        for m in metrics:
            entry["final"][m] = summarize_across_seeds([get(r, m) for r in rs])
        for thresh in (0.1, 0.01):
            steps = [threshold_step(r, "target_train_probe", thresh, common) for r in rs]
            entry["thresholds"]["target_probe_ce_le_%g" % thresh] = {
                "per_seed_steps": steps,
                "reached_by_all_seeds": all(s is not None for s in steps),
                "summary": summarize_across_seeds([s for s in steps if s is not None]),
            }
        entry["timing"]["train_seconds"] = summarize_across_seeds(
            [r["summary"]["timing"]["train_seconds"] for r in rs])
        arms_out[arm] = entry

    # Paired per-seed differences (accuracy in percentage points).
    pairs = {}
    for a, b in (("W", "A"), ("P", "A"), ("P", "W")):
        if not (by_arm.get(a) and by_arm.get(b)):
            continue
        per_seed = {}
        for m in metrics:
            diffs = []
            for ra in by_arm[a]:
                rb = next((x for x in by_arm[b] if x["seed"] == ra["seed"]), None)
                if rb is None:
                    continue
                d = get(ra, m) - get(rb, m)
                diffs.append(d * 100.0 if m.endswith("accuracy") else d)
            per_seed[m] = summarize_across_seeds(diffs)
        pairs["%s_minus_%s" % (a, b)] = per_seed

    agg = {
        "experiment": "exp1_gaussian_warmstart",
        "total_budget_updates": total,
        "common_checkpoints": common,
        "arms": arms_out,
        "paired_differences": pairs,
        "units": {"accuracy_differences": "percentage points", "ce_differences": "nats"},
        "notes": {
            "budget": ("all arms spend B updates in total; the shared sigma=1 prefix is "
                       "counted in both W and P even though it is computed once"),
            "thresholds": ("first COMMON checkpoint at which target training-probe CE "
                           "falls to the stated level, in global updates including "
                           "warm-start pretraining; not interpolated"),
            "uncertainty": ("three paired seeds support a preliminary comparison only; "
                            "overlapping error bars are not evidence of a difference"),
            "arm_A": "reused unchanged from Experiment 0 (results/exp0_gaussian)",
        },
    }
    out = Path(out_root) / "aggregate.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2, default=str)
    _write_csv(agg, by_arm, metrics, get, Path(out_root) / "aggregate.csv")
    return agg


def _write_csv(agg, by_arm, metrics, get, path) -> None:
    import csv
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "seed"] + metrics)
        for arm in ARMS:
            for r in by_arm.get(arm, []):
                w.writerow([arm, r["seed"]] + [get(r, m) for m in metrics])


# --------------------------------------------------------------------------
# Transition-invariance verification (spec section 3)
# --------------------------------------------------------------------------

def _eval_records(path):
    return [m for m in read_metrics(path) if m.get("record") == "eval"]


def verify_transition_invariance(out_root, seeds, warm: int, mid: int, tol: float = 1e-9) -> dict:
    """Target metrics must not move when only the active training sigma changes.

    Two kinds of transition are checked, both at *identical weights*:

    * the **branch** transition at update ``warm``, where the "before" record is
      the prefix run's final evaluation (active sigma = warm_sigma) and the
      "after" record is the arm's ``branch_start`` evaluation (active sigma =
      the arm's next level).  These live in different runs but share the
      checkpoint, so this is a genuine cross-run check.
    * the **in-run** transition at ``warm + mid`` for arm P, where the trainer
      logs an explicit ``pre_transition`` / ``post_transition`` pair.
    """
    out_root = Path(out_root)
    checks, failures = [], []

    for seed in seeds:
        pre_path = prefix_dir(out_root, seed) / "metrics.jsonl"
        if not pre_path.exists():
            continue
        pre_recs = [m for m in _eval_records(pre_path) if m["step"] == warm]
        if not pre_recs:
            continue
        pre = pre_recs[-1]
        for arm in ("W", "P"):
            m_path = arm_dir(out_root, arm, seed) / "metrics.jsonl"
            if not m_path.exists():
                continue
            post = next((m for m in _eval_records(m_path)
                         if m["step"] == warm and m.get("phase") == "branch_start"), None)
            if post is None:
                continue
            row = {"kind": "branch", "seed": seed, "arm": arm, "step": warm,
                   "sigma_before": pre["parameter"], "sigma_after": post["parameter"],
                   "deltas": {}, "ok": True}
            for block in ("target_train_probe", "target_val"):
                for key in ("ce", "accuracy"):
                    d = abs(pre[block][key] - post[block][key])
                    row["deltas"]["%s.%s" % (block, key)] = d
                    if d > tol:
                        row["ok"] = False
            if not row["ok"]:
                failures.append(row)
            checks.append(row)

        m_path = arm_dir(out_root, "P", seed) / "metrics.jsonl"
        if m_path.exists():
            recs = _eval_records(m_path)
            step = warm + mid
            a = next((m for m in recs if m["step"] == step
                      and m.get("phase") == "pre_transition"), None)
            b = next((m for m in recs if m["step"] == step
                      and m.get("phase") == "post_transition"), None)
            if a and b:
                row = {"kind": "in_run", "seed": seed, "arm": "P", "step": step,
                       "sigma_before": a["parameter"], "sigma_after": b["parameter"],
                       "deltas": {}, "ok": True}
                for block in ("target_train_probe", "target_val"):
                    for key in ("ce", "accuracy"):
                        d = abs(a[block][key] - b[block][key])
                        row["deltas"]["%s.%s" % (block, key)] = d
                        if d > tol:
                            row["ok"] = False
                if not row["ok"]:
                    failures.append(row)
                checks.append(row)

    report = {
        "tolerance": tol,
        "n_checks": len(checks),
        "all_passed": not failures,
        "failures": failures,
        "checks": checks,
        "meaning": ("original-image (target) metrics are computed at the family's target "
                    "endpoint and must be independent of the level currently being "
                    "trained on; any non-zero delta would indicate that target "
                    "evaluation is coupled to the active training transform"),
    }
    with open(out_root / "transition_invariance.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return report

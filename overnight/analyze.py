"""Collect, verify and tabulate the follow-up (new 160-epoch runs + the 42 historical endpoints).

    py -m overnight.analyze

Inputs: ``raw/<tag>/<account>/ovn`` (pulled kernel outputs; the latest tag with ``DONE.json`` wins
for a run), ``raw/eval/maxmonstre/ovn/old`` (P0-P3 on the 42 30-epoch endpoints) and the previous
study's ``results/per_run.csv`` (reused P0/P1 values).  Outputs go to ``analysis/``.
Accuracy differences are percentage points; CE differences are nats; SD of a gain = sample SD of the
seed-paired differences.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import matrix as MX
from .bnpolicies import POLICIES

RAW = MX.STUDY / "raw"
OUT = MX.STUDY / "analysis"
OLD_RESULTS = MX.PREV_STUDY_WORKTREE / "results" / "per_run.csv"
HIST = {"sgd": "historical_sgd_30", "adamw": "historical_adamw_30"}
CAMPAIGNS = ("historical_sgd_30", "historical_adamw_30") + MX.REGIMES
CAMPAIGN_LABEL = {"historical_sgd_30": "SGD lr 0.005, cosine, no aug, 30 ep (historical)",
                  "historical_adamw_30": "AdamW lr 0.02, cosine, no aug, 30 ep (historical)",
                  "sgd_standard_aug_160": "SGD lr 0.1, steps 80/120, crop+flip, 160 ep",
                  "adamw_long_noaug_160": "AdamW lr 0.02, cosine, no aug, 160 ep",
                  "adamw_long_aug_160": "AdamW lr 0.02, cosine, crop+flip, 160 ep"}
POLICY_SHORT = {"P0_saved_native": "P0", "P1_cumulative_clean_500": "P1", "P2_cumulative_clean_32": "P2",
                "P3_ema_training_loader": "P3"}
CONTRASTS = [("%s - Plain" % MX.SHORT[a], a, "plain") for a in MX.ARMS if a != "plain"] + [
    ("R - SDPoint", "resolution_max_b1", "sdpoint"), ("RG - SDPoint", "resolution_max_b1_gaussian_conv", "sdpoint"),
    ("G - CBS-pub", "gaussian_postrelu", "cbs_published_schedule"), ("G - CBS-bm", "gaussian_postrelu", "cbs_budget_matched"),
    ("RG - R", "resolution_max_b1_gaussian_conv", "resolution_max_b1"),
    ("CBS-bm - CBS-pub", "cbs_budget_matched", "cbs_published_schedule")]


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tags():
    return sorted(p.name for p in RAW.iterdir() if p.is_dir() and p.name not in ("pilot", "eval"))


def collect_new():
    """run -> dict(dir, tag, account) for every completed new cell (latest tag wins)."""
    done, seen = {}, {}
    for tag in _tags():
        for d in sorted((RAW / tag).glob("*/ovn/cells/*")):
            acc = d.parts[-4]
            seen.setdefault(d.name, []).append({"tag": tag, "account": acc, "dir": d,
                                                "done": (d / "DONE.json").exists(), "failed": (d / "FAILED.json").exists()})
            if (d / "DONE.json").exists():
                done[d.name] = {"tag": tag, "account": acc, "dir": d}
    return done, seen


def verify_manifests():
    rep = []
    for m in sorted(RAW.glob("*/*/ovn/MANIFEST.json")):
        root = m.parent
        listed = json.loads(m.read_text())
        missing = [r for r in listed if not (root / r).exists()]
        bad = [r for r in listed if (root / r).exists() and sha_file(root / r) != listed[r]["sha256"]]
        rep.append({"tag": root.parts[-3], "account": root.parts[-2], "files_listed": len(listed), "missing": len(missing),
                    "hash_mismatch": len(bad), "complete_json": (root / "COMPLETE.json").exists(),
                    "missing_examples": missing[:5], "mismatch_examples": bad[:5]})
    return pd.DataFrame(rep)


def rows_new(done):
    rows, curves_saved, curves_p1, costs, checks = [], [], [], [], []
    for run, info in sorted(done.items()):
        regime, arm, seed = run.split("__")
        seed = int(seed[4:])
        d = info["dir"]
        pol = json.loads((d / "endpoint_policies.json").read_text())
        dn = json.loads((d / "DONE.json").read_text())
        summ = json.loads((d / "summary.json").read_text())
        for p, rec in pol["policies"].items():
            for split in ("train_full", "test_full"):
                rows.append({"campaign": regime, "arm": arm, "seed": seed, "policy": POLICY_SHORT[p], "split": split.split("_")[0],
                             "acc": rec[split]["acc"], "ce": rec[split]["ce"], "n": rec[split]["n"], "source": "new run, final checkpoint epoch 160",
                             "state": json.dumps(rec["state"], sort_keys=True), "buffers_sha256": rec["buffers_sha256"],
                             "checkpoint_sha256": pol["checkpoint_sha256"], "account": info["account"], "tag": info["tag"]})
        ck_file = d / "checkpoints" / "epoch_160.pt"
        checks.append({"campaign": regime, "arm": arm, "seed": seed, "account": info["account"], "tag": info["tag"],
                       "updates": summ["updates"], "updates_ok": summ["updates"] == MX.U,
                       "checkpoint_present": ck_file.exists(),
                       "checkpoint_sha_matches": ck_file.exists() and sha_file(ck_file) == pol["checkpoint_sha256"],
                       "recorded_check_matches": dn["recorded_check_matches"], "P0_repeat_identical": dn["P0_repeat_identical"],
                       "learned_unchanged_all_policies": all(r["learned_tensors_unchanged"] for r in pol["policies"].values()),
                       "config_sha_frozen": dn["config_sha256"] == json.loads((MX.ROOT / "overnight/frozen_hashes.json").read_text())["config_sha256"][run],
                       "extended_perms_sha_frozen": dn["extended_perms_sha256"] == json.loads((MX.ROOT / "overnight/frozen_hashes.json").read_text())["extended_perms_sha256"][str(seed)],
                       "finite_final_train_loss": bool(np.isfinite(json.loads((d / "metrics.json").read_text())[-1]["train_loss_epoch"])),
                       "native_state": json.dumps(summ["native_inference_state"], sort_keys=True)})
        for m in json.loads((d / "metrics.json").read_text()):
            su = (m.get("state_used") or {}).get("state") or {}
            rec = {"campaign": regime, "arm": arm, "seed": seed, "epoch": m["epoch"], "update": m["update"],
                   "lr_last_update": m["lr_last_update"], "online_train_loss_epoch": m["train_loss_epoch"],
                   "state_resolution": su.get("resolution"), "state_level": su.get("sigma"),
                   "per_site_sigma_max": max((m.get("state_used") or {}).get("per_site_sigma") or [0.0]) if m.get("state_used") else None,
                   "bn": "saved running statistics"}
            for path in ("current", "target"):
                for split in ("train_probe", "test"):
                    e = m["eval"][path][split]
                    rec["%s_%s_acc" % (path, "probe500" if split == "train_probe" else "test")] = e["acc"]
                    rec["%s_%s_ce" % (path, "probe500" if split == "train_probe" else "test")] = e["ce"]
            curves_saved.append(rec)
        for c in json.loads((d / "p1_curve.json").read_text()):
            curves_p1.append({"campaign": regime, "arm": arm, "seed": seed, "epoch": c["epoch"], "update": c["update"],
                              "evaluated_state": json.dumps(c["state"], sort_keys=True), "state_note": c["evaluated_state"],
                              "per_site_sigma_max": max(c["per_site_sigma"] or [0.0]),
                              "train_acc": c["train_full"]["acc"], "train_ce": c["train_full"]["ce"],
                              "test_acc": c["test_full"]["acc"], "test_ce": c["test_full"]["ce"], "bn": "P1"})
        t = json.loads((d / "timing.json").read_text())
        costs.append({"campaign": regime, "arm": arm, "seed": seed, "gpu": t["gpu"], "account": info["account"],
                      "training_loop_seconds": t["training"]["train_seconds"], "training_gpu_hours": t["training"]["train_seconds"] / 3600,
                      "periodic_saved_bn_eval_seconds": t["periodic_saved_bn_evaluation_seconds"],
                      "p1_curve_diagnostic_seconds": t["p1_curve_diagnostic_seconds"],
                      "final_policy_evaluation_seconds": t["final_policy_evaluation_seconds"],
                      **{"final_%s_seconds" % POLICY_SHORT[k]: v for k, v in t["final_policy_seconds_each"].items()},
                      "run_wall_seconds_total": t["training"]["wall_seconds"] + t["final_policy_evaluation_seconds"],
                      "peak_cuda_memory_allocated_mib": max(s.get("peak_cuda_memory_allocated_mib", 0) for s in t["sessions"]),
                      "sessions": len(t["sessions"]), "concurrency": t["concurrency"],
                      "timing_scope": "training_loop = wall of Trainer.run minus periodic evaluation minus P1-curve diagnostics; includes checkpoint writes"})
    return rows, curves_saved, curves_p1, costs, checks


def rows_old():
    """Reused P0/P1 (previous study) + new P2/P3 and recomputation checks (evaluation job)."""
    rows, checks = [], []
    old = pd.read_csv(OLD_RESULTS)
    evdirs = {p.parent.name: p.parent for p in RAW.glob("eval/*/ovn/old/*/DONE.json")}
    for _, r in old.iterrows():
        camp = HIST[r.setting]
        base = {"campaign": camp, "arm": r.arm, "seed": int(r.seed), "state": r.native_state, "checkpoint_sha256": r.checkpoint_sha256,
                "account": r.account, "tag": "comparison-cbs-sdpoint"}
        for pol, col in (("P0", "saved_native"), ("P1", "panelA")):
            for split in ("train", "test"):
                rows.append({**base, "policy": pol, "split": split, "acc": r["%s_%s_full_acc" % (col, split)], "ce": r["%s_%s_full_ce" % (col, split)],
                             "n": 50000 if split == "train" else 10000, "source": "reused from comparison-cbs-sdpoint (%s)" % col,
                             "buffers_sha256": None})
        d = evdirs.get(r.run)
        chk = {"campaign": camp, "arm": r.arm, "seed": int(r.seed), "evaluated": d is not None}
        if d is not None:
            pol = json.loads((d / "endpoint_policies.json").read_text())
            for p in ("P2_cumulative_clean_32", "P3_ema_training_loader"):
                rec = pol["policies"][p]
                for split in ("train_full", "test_full"):
                    rows.append({**base, "policy": POLICY_SHORT[p], "split": split.split("_")[0], "acc": rec[split]["acc"], "ce": rec[split]["ce"],
                                 "n": rec[split]["n"], "source": "new evaluation of the historical final checkpoint", "buffers_sha256": rec["buffers_sha256"],
                                 "account": "maxmonstre", "tag": "eval"})
            chk.update({"checkpoint_identity_matches": pol["checkpoint_identity_matches_old_panels"] and pol["checkpoint_sha256"] == r.checkpoint_sha256,
                        "native_state_matches": pol["native_state_matches_old_panels"], "reuse_checks_match": pol["all_reuse_checks_match"],
                        "max_abs_diff_acc_P0_P1": max(v["abs_diff_acc"] for v in pol["reuse_check"].values()),
                        "max_abs_diff_ce_P0_P1": max(v["abs_diff_ce"] for v in pol["reuse_check"].values()),
                        "P0_repeat_identical": pol["P0_repeat_identical"],
                        "learned_unchanged_all_policies": all(v["learned_tensors_unchanged"] for v in pol["policies"].values()),
                        "P3_batch": pol["physical_batch_for_P3"]})
        checks.append(chk)
    return rows, checks


def paired(df):
    out = []
    for (camp, pol, split), g in df.groupby(["campaign", "policy", "split"]):
        piv_acc = g.pivot_table(index="seed", columns="arm", values="acc")
        piv_ce = g.pivot_table(index="seed", columns="arm", values="ce")
        for name, a, b in CONTRASTS:
            if a not in piv_acc or b not in piv_acc:
                continue
            both = piv_acc[[a, b]].dropna().index
            if len(both) == 0:
                continue
            da = 100 * (piv_acc.loc[both, a] - piv_acc.loc[both, b])
            dc = piv_ce.loc[both, a] - piv_ce.loc[both, b]
            rec = {"campaign": camp, "policy": pol, "split": split, "contrast": name, "arm_a": a, "arm_b": b, "n_pairs": len(both),
                   "seeds": ",".join(map(str, both))}
            for s in MX.SEEDS:
                rec["acc_diff_seed%d_pp" % s] = da.get(s, np.nan)
            rec.update({"acc_diff_mean_pp": da.mean(), "acc_diff_sd_pp": da.std(ddof=1) if len(da) > 1 else np.nan,
                        "acc_n_positive": int((da > 0).sum()), "acc_n_negative": int((da < 0).sum()), "acc_n_zero": int((da == 0).sum())})
            for s in MX.SEEDS:
                rec["ce_diff_seed%d" % s] = dc.get(s, np.nan)
            rec.update({"ce_diff_mean": dc.mean(), "ce_diff_sd": dc.std(ddof=1) if len(dc) > 1 else np.nan,
                        "ce_n_lower": int((dc < 0).sum()), "ce_n_higher": int((dc > 0).sum())})
            out.append(rec)
    return pd.DataFrame(out)


def summary(df):
    out = []
    for (camp, pol, split, arm), g in df.groupby(["campaign", "policy", "split", "arm"]):
        g = g.set_index("seed")
        rec = {"campaign": camp, "campaign_label": CAMPAIGN_LABEL[camp], "policy": pol, "split": split, "arm": arm, "arm_short": MX.SHORT[arm],
               "n_seeds": len(g), "acc_mean_pct": 100 * g.acc.mean(), "acc_sd_pct": 100 * g.acc.std(ddof=1) if len(g) > 1 else np.nan,
               "ce_mean": g.ce.mean(), "ce_sd": g.ce.std(ddof=1) if len(g) > 1 else np.nan}
        for s in MX.SEEDS:
            rec["acc_seed%d_pct" % s] = 100 * g.acc.get(s, np.nan)
            rec["ce_seed%d" % s] = g.ce.get(s, np.nan)
        out.append(rec)
    return pd.DataFrame(out)


def bn_sensitivity(df):
    out = []
    t = df[df.split == "test"].pivot_table(index=["campaign", "arm", "seed"], columns="policy", values=["acc", "ce"])
    for (camp, arm, seed), r in t.iterrows():
        rec = {"campaign": camp, "arm": arm, "seed": seed}
        for p in ("P0", "P2", "P3"):
            if ("acc", p) in r and ("acc", "P1") in r:
                rec["test_acc_%s_minus_P1_pp" % p] = 100 * (r[("acc", p)] - r[("acc", "P1")])
                rec["test_ce_%s_minus_P1" % p] = r[("ce", p)] - r[("ce", "P1")]
        out.append(rec)
    return pd.DataFrame(out)


def ordering(pairs):
    sel = pairs[(pairs.split == "test") & pairs.contrast.isin(["R - SDPoint", "RG - SDPoint"])]
    out = []
    for (camp, name), g in sel.groupby(["campaign", "contrast"]):
        g = g.set_index("policy")
        rec = {"campaign": camp, "contrast": name}
        for p in ("P0", "P1", "P2", "P3"):
            if p in g.index:
                rec["%s_mean_pp" % p] = g.loc[p, "acc_diff_mean_pp"]
                rec["%s_signs(+/-)" % p] = "%d/%d" % (g.loc[p, "acc_n_positive"], g.loc[p, "acc_n_negative"])
        means = [rec[k] for k in rec if k.endswith("_mean_pp")]
        rec["range_across_policies_pp"] = max(means) - min(means) if means else np.nan
        rec["sign_of_mean_changes_across_policies"] = len({np.sign(m) for m in means}) > 1 if means else None
        out.append(rec)
    return pd.DataFrame(out)


def cross_regime(df, provenance):
    """Gain over Plain per seed, P1 and P0 test accuracy, along the AdamW sequence."""
    out = []
    seq = [("historical_adamw_30", "adamw_long_noaug_160"), ("adamw_long_noaug_160", "adamw_long_aug_160"),
           ("historical_sgd_30", "sgd_standard_aug_160")]
    for pol in ("P1", "P0", "P2", "P3"):
        t = df[(df.split == "test") & (df.policy == pol)].pivot_table(index=["campaign", "seed"], columns="arm", values="acc")
        for a_camp, b_camp in seq:
            for arm in MX.ARMS:
                if arm == "plain":
                    continue
                rec = {"policy": pol, "from": a_camp, "to": b_camp, "arm": arm,
                       "pairing": provenance.get((a_camp, b_camp), "")}
                ga, gb = {}, {}
                for s in MX.SEEDS:
                    try:
                        ga[s] = 100 * (t.loc[(a_camp, s), arm] - t.loc[(a_camp, s), "plain"])
                        gb[s] = 100 * (t.loc[(b_camp, s), arm] - t.loc[(b_camp, s), "plain"])
                    except KeyError:
                        continue
                ga = {k: v for k, v in ga.items() if np.isfinite(v)}
                gb = {k: v for k, v in gb.items() if np.isfinite(v)}
                common = sorted(set(ga) & set(gb))
                d = np.array([gb[s] - ga[s] for s in common])
                rec.update({"n_pairs": len(common), **{"gain_from_seed%d_pp" % s: ga.get(s, np.nan) for s in MX.SEEDS},
                            **{"gain_to_seed%d_pp" % s: gb.get(s, np.nan) for s in MX.SEEDS},
                            "gain_from_mean_pp": np.mean(list(ga.values())) if ga else np.nan,
                            "gain_to_mean_pp": np.mean(list(gb.values())) if gb else np.nan,
                            "change_mean_pp": d.mean() if len(d) else np.nan, "change_sd_pp": d.std(ddof=1) if len(d) > 1 else np.nan,
                            "change_n_positive": int((d > 0).sum()), "change_n_negative": int((d < 0).sum())})
                out.append(rec)
    return pd.DataFrame(out)


def status_manifest(seen, done):
    alloc_p = MX.STUDY / "protocol" / "ALLOCATION.json"
    alloc = json.loads(alloc_p.read_text()) if alloc_p.exists() else {"accounts": {}}
    where = {}
    for acc, v in alloc["accounts"].items():
        for g, q in enumerate(v["queues"]):
            for r in q:
                where[r] = (acc, g)
    launches = {}
    for p in sorted(RAW.glob("launch_*.json")):
        j = json.loads(p.read_text())
        launches.setdefault(j["account"], []).append(j)
    fz = json.loads((MX.ROOT / "overnight/frozen_hashes.json").read_text())
    rows = []
    for c in MX.cells():
        r = c["run"]
        hist = seen.get(r, [])
        st = "done" if r in done else ("failed" if any(h["failed"] for h in hist) else
                                       "incomplete (restartable)" if hist else "not started / not collected")
        acc, g = where.get(r, (None, None))
        env = None
        if r in done:
            env = json.loads((done[r]["dir"] / "environment.json").read_text()).get("gpu")
        rows.append({"regime": c["regime"], "arm": c["arm"], "seed": c["seed"], "priority": c["priority"], "run": r,
                     "config_sha256": fz["config_sha256"][r], "allocated_account": acc, "allocated_gpu_queue": g,
                     "completed_account": done[r]["account"] if r in done else None,
                     "kaggle_kernel": "%s (tag %s)" % (done[r]["account"], done[r]["tag"]) if r in done else
                     ";".join("%s/%s" % (h["account"], h["tag"]) for h in hist) or None,
                     "gpu_type": env, "status": st,
                     "attempts": ";".join("%s:%s" % (h["tag"], "done" if h["done"] else "failed" if h["failed"] else "partial") for h in hist),
                     "output_location": str(done[r]["dir"].relative_to(MX.STUDY)).replace("\\", "/") if r in done else None})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    done, seen = collect_new()
    new_rows, curves_saved, curves_p1, costs, new_checks = rows_new(done)
    old_rows, old_checks = rows_old()
    df = pd.DataFrame(new_rows + old_rows)
    df["campaign_label"] = df.campaign.map(CAMPAIGN_LABEL)
    df["arm_short"] = df.arm.map(MX.SHORT)
    df.to_csv(OUT / "per_run_policy_long.csv", index=False)
    wide = df.pivot_table(index=["campaign", "arm", "seed"], columns=["policy", "split"], values=["acc", "ce"])
    wide.columns = ["%s_%s_%s" % (p, s, m) for m, p, s in wide.columns]
    wide.reset_index().to_csv(OUT / "per_run_wide.csv", index=False)
    pairs = paired(df)
    pairs.to_csv(OUT / "paired_contrasts.csv", index=False)
    summary(df).to_csv(OUT / "summary_by_arm.csv", index=False)
    bn_sensitivity(df).to_csv(OUT / "bn_policy_sensitivity_per_run.csv", index=False)
    ordering(pairs).to_csv(OUT / "r_rg_vs_sdpoint_across_policies.csv", index=False)
    prov = {("historical_adamw_30", "adamw_long_noaug_160"): "same init_seed<k> and perm_seed<k> epochs 0-29; different horizon, LR schedule length and intervention schedules; separate campaigns",
            ("adamw_long_noaug_160", "adamw_long_aug_160"): "same init, same 160-epoch order, same schedules; differs only by crop/flip",
            ("historical_sgd_30", "sgd_standard_aug_160"): "same init and first-30-epoch order; different optimizer settings, schedule, horizon and augmentation (descriptive only)"}
    cross_regime(df, prov).to_csv(OUT / "gain_changes_across_regimes.csv", index=False)
    pd.DataFrame(curves_saved).to_csv(OUT / "learning_curves_saved_bn_per_epoch.csv", index=False)
    pd.DataFrame(curves_p1).to_csv(OUT / "learning_curves_p1.csv", index=False)
    pd.DataFrame(costs).to_csv(OUT / "costs.csv", index=False)
    pd.DataFrame(new_checks).to_csv(OUT / "checks_new_runs.csv", index=False)
    pd.DataFrame(old_checks).to_csv(OUT / "checks_historical_endpoints.csv", index=False)
    man = verify_manifests()
    man.to_csv(OUT / "download_manifest_verification.csv", index=False)
    stat = status_manifest(seen, done)
    stat.to_csv(OUT / "status_manifest.csv", index=False)
    print("new done %d/63, old evaluated %d/42, rows %d, manifest problems %d"
          % (len(done), sum(c["evaluated"] for c in old_checks), len(df),
             int((man.missing + man.hash_mismatch).sum()) if len(man) else -1))
    print(stat.status.value_counts().to_dict())


if __name__ == "__main__":
    main()

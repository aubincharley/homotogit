"""Collect runs across Kaggle accounts and check which ones are actually paired.

Campaign comparisons are only meaningful between runs that started from the same
initial weights and BatchNorm buffers and saw the same data in the same order.
Seed numbers do not prove that; digests do.  For every run this reads the study
directory's own ``shared_indices.npz`` and ``init_seed<k>.pt`` and hashes them,
so pairing is checked from the artifacts rather than trusted.

Runs that share a *pairing key* -- (subset, train probe, permutation/order for
that seed, initial weights for that seed) -- may be compared as paired.  Runs
that do not must be labelled contextual.  Two accounts running the same
configuration on the same Kaggle image normally produce the same key; a
different image or GPU type can break it, which is exactly what this catches.

Usage::

    py scripts/aggregate_runs.py                       # every downloaded run
    py scripts/aggregate_runs.py --glob 'progres-*'    # a subset
    py scripts/aggregate_runs.py --json out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "results" / "kaggle_outputs"


def _sha_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _sha_state(path: Path) -> str:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(np.ascontiguousarray(sd[k].cpu().numpy()).tobytes())
    return h.hexdigest()


def study_digests(study_dir: Path) -> dict:
    """Digests of the shared state a study generated, or {} if unavailable."""
    out, npz = {}, study_dir / "shared_indices.npz"
    if npz.is_file():
        z = np.load(npz)
        for key in z.files:
            out[key] = _sha_array(z[key])
    for init in sorted(study_dir.glob("init_seed*.pt")):
        out[init.stem] = _sha_state(init)
    return out


def collect(pattern: str = "*") -> list:
    """One record per training run found under results/kaggle_outputs."""
    records = []
    for job in sorted(OUTPUTS.glob(pattern)):
        if not job.is_dir():
            continue
        launch = json.loads((job / "launch.json").read_text()) \
            if (job / "launch.json").is_file() else {}
        for study in sorted(p for p in job.iterdir() if p.is_dir() and p.name != "_repo"):
            env = json.loads((study / "environment.json").read_text()) \
                if (study / "environment.json").is_file() else {}
            digests = study_digests(study)
            shipped = json.loads((study / "pairing_verification.json").read_text()) \
                if (study / "pairing_verification.json").is_file() else None
            for run in sorted(p for p in study.iterdir() if p.is_dir()):
                summ = run / "summary.json"
                if not summ.is_file():
                    continue
                s = json.loads(summ.read_text())
                seed = s.get("seed")
                key_parts = [digests.get("subset"), digests.get("train_probe"),
                             digests.get("perm_seed%s" % seed)
                             or digests.get("order_seed%s" % seed),
                             digests.get("init_seed%s" % seed)]
                records.append({
                    "job": job.name, "study": study.name, "label": s.get("label"),
                    "account": launch.get("account", "maxnicaise (pre-accounts)"),
                    "kernel_id": launch.get("kernel_id"),
                    "seed": seed, "kind": s.get("kind"),
                    "epochs": s.get("epochs"), "updates": s.get("updates"),
                    "n_train": s.get("n_train"), "n_test": s.get("n_test"),
                    "test_acc": s.get("final_test_acc", s.get("final_val_acc")),
                    "test_ce": s.get("final_test_ce", s.get("final_val_ce")),
                    "wall_seconds": s.get("wall_seconds"),
                    "eval_overhead_s": s.get("eval_overhead_s"),
                    "peak_mem_mib": s.get("peak_mem_mib"),
                    "gpu": s.get("gpu"), "torch": env.get("torch"),
                    "pairing_key": None if any(v is None for v in key_parts)
                    else hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:16],
                    "pairing_parts_missing": [n for n, v in zip(
                        ("subset", "train_probe", "order", "init"), key_parts)
                        if v is None],
                    "shipped_pairing_check": (shipped or {}).get("all_match"),
                    "path": str(run.relative_to(ROOT)),
                })
    return records


def paired_groups(records: list) -> dict:
    groups = {}
    for r in records:
        groups.setdefault(r["pairing_key"], []).append(r)
    return groups


def require_paired(records: list, labels=None):
    """Raise unless every named run shares one pairing key.

    Call this before plotting or differencing arms from different jobs or
    accounts; it turns an unpaired comparison into an error instead of a
    quietly wrong number.
    """
    sel = [r for r in records if labels is None or r["label"] in labels]
    keys = {r["pairing_key"] for r in sel}
    if None in keys:
        missing = [(r["label"], r["pairing_parts_missing"]) for r in sel
                   if r["pairing_key"] is None]
        raise SystemExit("cannot verify pairing (artifacts absent): %s" % missing)
    if len(keys) != 1:
        by = {}
        for r in sel:
            by.setdefault(r["pairing_key"], []).append(r["label"])
        raise SystemExit("runs are NOT paired; label these comparisons "
                         "contextual: %s" % by)
    return sel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="*", help="job-directory glob")
    ap.add_argument("--json", type=Path, help="write the full records here")
    args = ap.parse_args()

    records = collect(args.glob)
    if not records:
        raise SystemExit("no runs found under %s matching %r" % (OUTPUTS, args.glob))

    groups = paired_groups(records)
    print("%-28s %-22s %-11s %-5s %-8s %-8s %8s  %s"
          % ("run", "job", "account", "seed", "test acc", "test CE", "wall s", "pair"))
    print("-" * 118)
    for key, rs in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0] or "")):
        for r in rs:
            print("%-28s %-22s %-11s %-5s %-8s %-8s %8s  %s"
                  % (r["label"], r["job"][:22], r["account"][:11],
                     r["seed"], "%.4f" % r["test_acc"] if r["test_acc"] else "-",
                     "%.4f" % r["test_ce"] if r["test_ce"] else "-",
                     "%.0f" % r["wall_seconds"] if r["wall_seconds"] else "-",
                     key or "UNVERIFIED"))
        print()

    print("pairing groups: %d (runs sharing a group may be compared as paired)"
          % len([k for k in groups if k]))
    for key, rs in groups.items():
        if key is None:
            print("  UNVERIFIED  %d run(s) -- artifacts missing, treat as contextual"
                  % len(rs))
        else:
            accts = sorted({r["account"] for r in rs})
            torches = sorted({r["torch"] for r in rs if r["torch"]})
            print("  %s  %2d run(s)  accounts=%s  torch=%s"
                  % (key, len(rs), ",".join(accts), ",".join(torches)))

    if args.json:
        args.json.write_text(json.dumps(records, indent=2))
        print("wrote", args.json)


if __name__ == "__main__":
    main()

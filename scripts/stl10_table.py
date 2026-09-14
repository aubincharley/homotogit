"""Collect STL-10 run summaries into the benchmark table.

Reads every ``summary.json`` under the given roots (a Kaggle kernel output tree
works as-is) and prints mean +- sd of the final test accuracy per method, in the
shape of the CIFAR-10 table in the README, alongside the CIFAR-10 reference so
the two settings can be read side by side.

    py scripts/stl10_table.py runs/ --fetch     # --fetch pulls the kernels first
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from continuation_core.methods import METHODS as REGISTRY                # noqa: E402
import stl10_configs as S                                               # noqa: E402

LABEL = {"plain": "plain (control)",
         "resolution_max_b1": "resolution_max_b1",
         "gaussian_postrelu": "gaussian_postrelu",
         "resolution_max_b1_gaussian_conv": "resolution_max_b1_gaussian_conv"}


def mean_sd(values):
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, None
    return m, (sum((v - m) ** 2 for v in values) / (n - 1)) ** 0.5


def fetch(user: str, out: Path, seeds, methods) -> None:
    for seed in seeds:
        for method in methods:
            slug = "stl10-%s-seed%d" % (method.replace("_", "-"), seed)
            d = out / slug
            if (d / ".fetched").exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(["kaggle", "kernels", "output", "%s/%s" % (user, slug),
                                "-p", str(d)], capture_output=True, text=True)
            if list(d.rglob("summary.json")):
                (d / ".fetched").touch()
            else:
                print("  no output yet for %s" % slug, file=sys.stderr)


def collect(roots) -> dict:
    runs = {}
    for root in roots:
        for p in Path(root).rglob("summary.json"):
            s = json.loads(p.read_text())
            if s.get("dataset") != "stl10":
                continue
            method = s["method"].replace("__transfer", "")
            # a rerun of the same seed replaces the earlier one
            runs.setdefault(method, {})[s["seed"]] = {
                "acc": s["final"]["current"]["test"]["acc"],
                "probe": s["final"]["current"]["train_probe"]["acc"],
                "minutes": s["timing"]["wall_seconds"] / 60,
                "is_target": s["final_state_is_target"],
                "commit": (s["provenance"].get("code") or "")[:12],
            }
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="*", default=["runs"])
    ap.add_argument("--fetch", action="store_true", help="pull kernel outputs first")
    ap.add_argument("--user", default="aubincharley")
    ap.add_argument("--out", default="runs/stl10", help="where --fetch puts them")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.fetch:
        fetch(args.user, Path(args.out), seeds, S.METHODS)
        args.roots = list(args.roots) + [args.out]

    runs = collect(args.roots)
    if not runs:
        raise SystemExit("no STL-10 summary.json under %s" % ", ".join(map(str, args.roots)))

    print("STL-10 / ResNet-20-BN, %d epochs = %d updates, seed(s) %s"
          % (S.EPOCHS, S.EPOCHS * 40, ",".join(map(str, seeds))))
    print("sigma x3, resolution %s (reference %d), boundaries x%d\n"
          % (list(S.STL_R.values), S.NATIVE_RESOLUTION, S.BOUNDARY_SCALE))
    plain = runs.get("plain", {})
    base = sum(r["acc"] for r in plain.values()) / len(plain) if plain else None

    print("%-33s %-17s %-8s %-17s" % ("method", "STL-10 test", "vs plain", "CIFAR-10 test"))
    print("-" * 78)
    order = sorted(runs, key=lambda k: -mean_sd([r["acc"] for r in runs[k].values()])[0])
    for method in order:
        accs = [runs[method][s]["acc"] for s in sorted(runs[method])]
        m, sd = mean_sd(accs)
        cell = "%.2f" % (100 * m) + (" +- %.2f" % (100 * sd) if sd is not None else
                                     " (%d seed)" % len(accs))
        src = REGISTRY[method].source
        ref = ("%.2f +- %.2f" % (100 * src["final_test_acc_mean"], 100 * src["final_test_acc_sd"])
               if src else "-")
        delta = "-" if base is None or method == "plain" else "%+.2f pp" % (100 * (m - base))
        print("%-33s %-17s %-8s %-17s" % (LABEL.get(method, method), cell, delta, ref))

    print("\nper seed:")
    for method in order:
        cells = " ".join("s%d %.4f" % (s, runs[method][s]["acc"]) for s in sorted(runs[method]))
        print("  %-33s %s" % (method, cells))

    bad = [(mth, s) for mth in runs for s in runs[mth] if not runs[mth][s]["is_target"]]
    print("\nall runs ended in the target state:", not bad if True else bad)
    if bad:
        print("  NOT in target state:", bad)
    commits = {r["commit"] for mth in runs for r in runs[mth].values()}
    print("code commit(s):", ", ".join(sorted(c for c in commits if c)) or "unknown")
    print("wall minutes:", ", ".join(
        "%s %.1f" % (mth, sum(r["minutes"] for r in runs[mth].values()) / len(runs[mth]))
        for mth in order))
    missing = [(mth, s) for mth in S.METHODS for s in seeds
               if s not in runs.get(mth, {})]
    if missing:
        print("\nstill missing %d run(s): %s" % (len(missing), missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())

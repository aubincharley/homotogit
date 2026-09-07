"""Turn per-seed rows into the baseline numbers.

The whole point of this file is that a baseline is a distribution, not a number.
Every headline metric gets a mean, a min..max range, and a standard error of the
mean, and the report states outright how large a difference has to be before it
means anything -- so the next change that beats this by 0.03% can be told that it
did not.

Every run writes into its own directory under runs/, never on top of the last
one, and records how it was launched alongside what it found. A baseline you
cannot re-run is not a baseline, and "I think that was the 20-epoch one" is how
a week of results becomes unusable.
"""
import csv
import json
import math
import os
import shlex
import sys
import time

HEADLINE = [
    ("test_acc_selected", "test acc", "{:.4f}"),
    ("test_err_selected", "test err %", "{:.3f}"),
    ("test_acc_final", "final acc", "{:.4f}"),
    ("train_acc", "train acc", "{:.4f}"),
    ("gen_gap", "gen gap", "{:+.4f}"),
    ("worst_class_acc", "worst cls", "{:.4f}"),
    ("wall_s", "wall s", "{:.0f}"),
]

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


def _out_dir():
    """/kaggle/working is the only writable path in a kernel, and is what gets
    attached to the run output."""
    return "/kaggle/working" if os.path.isdir("/kaggle/working") else "."


def run_dir(cfg, root=None):
    """A fresh directory for this run: runs/<stamp>_<config>_<arch>/.

    Created before training rather than after, so a crashed run still leaves its
    invocation on disk, and so the path can be printed for tailing.
    """
    base = os.path.join(root or _out_dir(), "runs")
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg['config']}_{cfg['arch']}"
    path = os.path.join(base, stem)
    # A scripted sweep can start two runs inside the same second; that must
    # append rather than land both in one directory.
    attempt = 0
    while os.path.exists(path):
        attempt += 1
        path = os.path.join(base, f"{stem}-{attempt}")
    os.makedirs(path)
    _point_latest_at(base, os.path.basename(path))
    return path


def _point_latest_at(base, name):
    """runs/latest -> the newest run, for `cat runs/latest/results.json`."""
    link = os.path.join(base, "latest")
    try:
        if os.path.islink(link):
            os.remove(link)
        elif os.path.exists(link):
            return                      # a real directory named latest: leave it
        os.symlink(name, link)
    except OSError:                                         # pragma: no cover
        pass                            # a filesystem without symlinks is fine


def dump_invocation(cfg, directory, filename="invocation.json"):
    """How this run was launched, next to what it produced.

    The resolved config alone does not say where a value came from -- a flag, an
    env var, or the YAML -- so the raw argv and the CIFAR_* environment are kept
    verbatim. `command` is paste-able to repeat the run.
    """
    payload = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_name": cfg["config"],
        "argv": sys.argv,
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "env": {k: v for k, v in sorted(os.environ.items())
                if k.startswith("CIFAR_")},
        "cwd": os.getcwd(),
        "on_kaggle": os.path.isdir("/kaggle/working"),
        "config": cfg,
    }
    try:
        import torch
        payload["torch"] = torch.__version__
        payload["device_name"] = (torch.cuda.get_device_name(0)
                                  if torch.cuda.is_available() else "cpu")
    except Exception:                                       # noqa: BLE001
        pass
    path = os.path.join(directory, filename)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def summarise(runs):
    """Mean / min / max / std / sem for every numeric metric across seeds."""
    out = {"n_seeds": len(runs), "seeds": [r["seed"] for r in runs]}
    keys = [k for k, v in runs[0].items()
            if isinstance(v, (int, float)) and k != "seed"]
    for key in keys:
        values = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
        values = [v for v in values if not math.isnan(v)]
        if not values:
            continue
        mean = sum(values) / len(values)
        entry = {"mean": mean, "min": min(values), "max": max(values),
                 "n": len(values)}
        if len(values) > 1:
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            entry["std"] = math.sqrt(var)
            entry["sem"] = math.sqrt(var / len(values))
        out[key] = entry
    return out


def print_report(cfg, runs, stats, test_n=10000):
    print("\n" + "=" * 78)
    print(f"CIFAR-10 BASELINE  --  {cfg['arch']}, {runs[0]['params']:,} params, "
          f"{cfg['epochs']} epochs, {stats['n_seeds']} seeds")
    print(f"config '{cfg['config']}': sgd lr={cfg['lr']} mom={cfg['momentum']} "
          f"wd={cfg['weight_decay']} bs={cfg['batch_size']} "
          f"{cfg['schedule']}+{cfg['warmup_epochs']}ep warmup, "
          f"augment={bool(cfg['augment'])}")
    print("=" * 78)

    print("\nper seed:")
    print("  " + "".join(f"{h:>12}" for h in ["seed"] + [n for _, n, _ in HEADLINE]))
    for row in runs:
        cells = [str(row["seed"])] + [fmt.format(row[key])
                                      for key, _, fmt in HEADLINE]
        print("  " + "".join(f"{c:>12}" for c in cells))

    print("\nmean +- sem  (min .. max):")
    for key, name, fmt in HEADLINE:
        s = stats.get(key)
        if not s:
            continue
        sem = f" +- {fmt.format(s['sem']).lstrip('+')}" if "sem" in s else ""
        print(f"  {name:>12}  {fmt.format(s['mean'])}{sem}"
              f"   ({fmt.format(s['min'])} .. {fmt.format(s['max'])})")

    # How large does a difference have to be to mean anything? Two independent
    # means differ significantly at ~2 * sqrt(2) * sem, and on 10k test images
    # there is also a hard binomial floor no amount of seeds gets under.
    acc = stats.get("test_acc_selected", {})
    if "sem" in acc and acc["sem"] > 0:
        mde = 2 * math.sqrt(2) * acc["sem"]
        p = acc["mean"]
        binom = math.sqrt(p * (1 - p) / test_n)
        floor = max(mde, 2 * binom)
        print("\nresolution of this baseline:")
        print(f"  seed-to-seed sem             {acc['sem']:.5f}  "
              f"({acc['sem'] * test_n:.1f} test images)")
        print(f"  binomial sem on n={test_n}      {binom:.5f}  "
              f"({binom * test_n:.1f} test images)")
        print(f"  smallest credible difference {floor:.5f}  "
              f"({floor * test_n:.1f} test images)")
        print(f"  -> a change must beat {acc['mean']:.4f} by more than "
              f"{floor:.4f} to have shown anything")

    per_class = runs[0].get("per_class")
    if per_class:
        print("\nper-class accuracy (seed "
              f"{runs[0]['seed']}, final epoch):")
        for name, value in zip(CLASSES, per_class, strict=True):
            print(f"  {name:>12}  {value:.4f}")

    selection = runs[0]["selection"]
    print(f"\nmodel selection: {selection}"
          + (f" (val_size={cfg['val_size']}, mean selected epoch "
             f"{stats['selected_epoch']['mean']:.1f})" if selection == "best_val"
             else " -- no validation split, so 'selected' == final epoch"))
    print(f"throughput: {stats['img_per_s']['mean']:,.0f} img/s mean")


def dump(payload, directory, filename="results.json"):
    """Always write the numbers somewhere durable.

    /kaggle/working is the only writable path in a kernel and is what gets
    attached to the run output, so a failed W&B sync never costs the results.
    """
    path = os.path.join(directory, filename)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nresults written to {path}")
    return path


def dump_history(history, directory, stem="history"):
    """Every epoch of every seed, as JSONL and as CSV.

    Two formats because they get used differently: the JSONL is appendable and
    survives a row growing a new key, the CSV opens in anything. Both exist so a
    W&B sync that never happens does not cost the curves.
    """
    if not history:
        return []
    columns = list(dict.fromkeys(k for row in history for k in row))

    jsonl_path = os.path.join(directory, f"{stem}.jsonl")
    with open(jsonl_path, "w") as fh:
        for row in history:
            fh.write(json.dumps(row, default=str) + "\n")

    csv_path = os.path.join(directory, f"{stem}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)

    print(f"history written to {jsonl_path} and {csv_path} "
          f"({len(history)} epoch rows)")
    return [jsonl_path, csv_path]

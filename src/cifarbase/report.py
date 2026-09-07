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
    # The homotopy coordinate, reported next to the accuracy it bought. An arm
    # without an accuracy number attached to a drift number is not comparable to
    # any other arm.
    ("d_final", "drift d_T", "{:.4f}"),
    ("lambda_final", "lambda_T", "{:.3g}"),
    ("wall_s", "wall s", "{:.0f}"),
]

# On CIFAR-10 an effect worth believing has to clear about this much. Stated as a
# constant so the report and the protocol cannot drift apart.
EFFECT_THRESHOLD = 0.003

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


def arm_dir(parent, name):
    """A subdirectory of the run for one arm, or the run itself when unnamed.

    Arms as subdirectories rather than sibling runs is what makes
    `analyze.py runs/latest` draw the comparison: find_runs() treats a folder
    whose children hold results.json as a set of runs to overlay.
    """
    if not name:
        return parent
    path = os.path.join(parent, name)
    os.makedirs(path, exist_ok=True)
    return path


def print_comparison(arms, baseline=None):
    """Every arm on one table, differences stated against a named baseline arm.

    A difference is only reported next to what would make it credible: two
    independent means separate at about 2*sqrt(2)*sem, and on 10k test images
    there is a binomial floor underneath that no number of seeds gets past. An
    arm that beats the reference by less than the floor is printed with a dash,
    because it has shown nothing and a number there would invite reading it.
    """
    if len(arms) < 2:
        return
    reference = next((a for a in arms if a["name"] == baseline), arms[0])
    ref_acc = reference["stats"].get("test_acc_selected", {})

    print("\n" + "=" * 78)
    print(f"ARM COMPARISON  --  {len(arms)} arms, reference "
          f"'{reference['name']}'")
    print("=" * 78)
    header = ["arm", "test acc", "+-sem", "vs ref", "credible?", "d_T", "lambda_T"]
    print("  " + "".join(f"{h:>13}" for h in header))

    for arm in arms:
        acc = arm["stats"].get("test_acc_selected", {})
        mean = acc.get("mean", float("nan"))
        sem = acc.get("sem", float("nan"))
        drift = arm["stats"].get("d_final", {}).get("mean", float("nan"))
        lam = arm["stats"].get("lambda_final", {}).get("mean", float("nan"))
        if arm is reference:
            delta, verdict = "--", "reference"
        else:
            gap = mean - ref_acc.get("mean", float("nan"))
            delta = f"{gap:+.4f}"
            verdict = _credible(gap, sem, ref_acc.get("sem"), mean)
        cells = [arm["name"][:13], f"{mean:.4f}",
                 "n/a" if sem != sem else f"{sem:.4f}",
                 delta, verdict, f"{drift:.3f}", f"{lam:.3g}"]
        print("  " + "".join(f"{c:>13}" for c in cells))

    print("\n  'credible?' is |difference| against max(2*sqrt(2)*sem_pooled, "
          "2*binomial sem).")
    print("  A 'no' means the arms are not separated by this experiment -- not "
          "that they are equal.")


def _credible(gap, sem_a, sem_b, mean, test_n=10000):
    if sem_a != sem_a or sem_b is None or sem_b != sem_b:
        return "1 seed"
    pooled = math.sqrt(sem_a ** 2 + sem_b ** 2)
    binom = math.sqrt(max(mean, 1e-9) * (1 - max(mean, 1e-9)) / test_n)
    floor = max(2 * math.sqrt(2) * pooled, 2 * binom)
    return "yes" if abs(gap) > floor else "no"


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
        # .get, because a run too short to reach anchor_reference_step never
        # captures K_0 and so has no drift column -- a legitimate state for a
        # smoke test, and not worth a KeyError in the reporting.
        cells = [str(row["seed"])] + [fmt.format(row.get(key, float("nan")))
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
        print(f"  (the protocol's threshold for a real effect is "
              f"{EFFECT_THRESHOLD:.4f}; this run resolves "
              f"{'better' if floor < EFFECT_THRESHOLD else 'WORSE'} than that)")

    per_class = runs[0].get("per_class")
    if per_class:
        print("\nper-class accuracy (seed "
              f"{runs[0]['seed']}, final epoch):")
        for name, value in zip(CLASSES, per_class, strict=True):
            print(f"  {name:>12}  {value:.4f}")

    _print_anchor(cfg, runs, stats)

    selection = runs[0]["selection"]
    print(f"\nmodel selection: {selection}"
          + (f" (val_size={cfg['val_size']}, mean selected epoch "
             f"{stats['selected_epoch']['mean']:.1f})" if selection == "best_val"
             else " -- no validation split, so 'selected' == final epoch"))
    print(f"throughput: {stats['img_per_s']['mean']:,.0f} img/s mean")


def _print_anchor(cfg, runs, stats):
    """The anchor's own block. Printed for every arm, the unanchored one included.

    The three monitor lines are the reason this is not folded into the headline
    table. A controller that saturated, or whose feedback sign flipped, still
    produces a smooth lambda trace and a plausible accuracy -- so the run has to
    say so itself, in words, next to the number it produced. Reading those off a
    CSV afterwards is exactly what does not happen.
    """
    first = runs[0]
    if "anchor_mode" not in first:
        return
    mode = first["anchor_mode"]
    print(f"\nanchor: mode={mode}")
    if mode == "off":
        print("  no penalty applied; drift was measured and logged only")
    else:
        print(f"  lambda clip  [{first['lambda_min']:.4g}, "
              f"{first['lambda_max']:.4g}]"
              + (f"   target d*(T) = {cfg['anchor_dmax']:.4f}, "
                 f"beta = {cfg['anchor_beta']:.4g}"
                 if mode == "adaptive" else ""))
    for key, label, fmt in (("d_final", "drift d_T", "{:.4f}"),
                            ("d_ntk_final", "  via NTK", "{:.4f}"),
                            ("a_ntk_final", "  alignment a_T", "{:.4f}"),
                            ("scale_ntk_final", "  kernel scale", "{:.4f}"),
                            ("d_feature_final", "  via features", "{:.4f}"),
                            ("scale_feature_final", "  feature scale", "{:.4f}"),
                            ("dist_rel_final", "||w-w0||/||w0||", "{:.4f}"),
                            ("lambda_final", "lambda at T", "{:.4g}")):
        entry = stats.get(key)
        if entry:
            print(f"  {label:>16}  {fmt.format(entry['mean'])}"
                  f"   ({fmt.format(entry['min'])} .. {fmt.format(entry['max'])})")

    clips = [r.get("clip_hits", 0) for r in runs]
    flips = [r.get("sign_flip_step", -1) for r in runs]
    sats = [r.get("saturation_step", -1) for r in runs]
    print(f"  clip hits per seed          {clips}")
    if any(f >= 0 for f in flips):
        print(f"  !! FEEDBACK SIGN FLIPPED at steps {flips} (-1 = never). Every "
              f"step after that is constant-lambda training: the controller was "
              f"frozen because d had started responding POSITIVELY to lambda.")
    if any(s >= 0 for s in sats):
        print(f"  !! DRIFT SIGNAL SATURATED at steps {sats} (-1 = never). The "
              f"controller had no authority left past that point; treat the tail "
              f"of the run as constant lambda, not as a tracked schedule.")
    if all(f < 0 for f in flips) and all(s < 0 for s in sats):
        print("  monitors clean: feedback sign held and the drift signal stayed live")
    # d^2 = scale^2 - 2 a scale + 1. When the scale term dominates, the drift the
    # controller regulated was mostly the kernel getting bigger rather than the
    # function class rotating, and the run has to say so next to the number.
    scale = stats.get("scale_ntk_final", {}).get("mean")
    align = stats.get("a_ntk_final", {}).get("mean")
    if scale is not None and align is not None and scale > 2.0:
        print(f"  !! the kernel's NORM grew {scale:.1f}x while its alignment with "
              f"K_0 is {align:.2f}. d is therefore mostly reporting scale, not "
              f"geometry. Read the alignment column, and consider "
              f"probe_signal=alignment (which steers by 1-a, the same quantity "
              f"with the scale divided out) before attributing this drift to "
              f"feature learning.")

    if max(clips, default=0) > 0 and mode == "adaptive":
        print("  a persistently clipped lambda means the drift target is "
              "unreachable at this beta -- a finding, not a bug, but it means "
              "d*(t) was not actually tracked. Check the d vs d* panel.")

    _print_groups(first, stats)


def _print_groups(first, stats):
    """One row per parameter group: what the allocation did and what it bought.

    Printed even with the allocation off, where every a_g is 0 by construction:
    the point of the table is then the OTHER columns -- how the drift, the
    displacement and the group's share of the kernel differ across depth under a
    single shared lambda. That is the comparison the adaptive arm has to beat, and
    it is unreadable if only one arm prints it.
    """
    groups = first.get("anchor_groups")
    if not groups:
        return
    alloc = first.get("anchor_alloc", "off")
    grouping = first.get("anchor_grouping", "stage")
    print(f"\nper group ({len(groups)}, grouping {grouping!r}), "
          f"allocation {alloc}:")
    print(f"  {'group':<10} {'||w0||^2':>10} {'lambda_T':>10} {'a_T':>8} "
          f"{'d_T':>9} {'|w-w0|/|w0|':>12} {'share K':>8} {'tau_T':>10}")
    w0_sq = first.get("anchor_w0_sq", {})
    for group in groups:
        def cell(key, fmt="{:>9.4f}", width=9):
            # The across-seed mean when summarise() aggregated the column, and
            # seed 0's own value when it did not (a single-seed arm, or a column
            # only some seeds produced).
            entry = stats.get(f"{key}_{group}_final")
            value = (entry["mean"] if entry is not None
                     else first.get(f"{key}_{group}_final"))
            return fmt.format(value) if value is not None else "-".rjust(width)
        print(f"  {group:<10} {w0_sq.get(group, float('nan')):>10.3g} "
              f"{cell('lam', '{:>10.4g}', 10)} "
              f"{cell('a', '{:>8.3f}', 8)} "
              f"{cell('d')} "
              f"{cell('dist_rel', '{:>12.5f}', 12)} "
              f"{cell('share', '{:>8.4f}', 8)} "
              f"{cell('tau', '{:>10.3g}', 10)}")

    spread = [first.get(f"a_{g}_final") for g in groups]
    spread = [v for v in spread if v is not None]
    if alloc == "adaptive" and spread:
        width = max(spread) - min(spread)
        print(f"  a spread {width:.3f} in log lambda ({math.exp(width):.1f}x "
              f"between the loosest and tightest group), sum "
              f"{sum(spread):+.2e} (must be ~0)")
        skipped = first.get("alloc_skipped", 0)
        if skipped:
            print(f"  !! {skipped} slow-loop updates were SKIPPED on a "
                  f"non-finite tension. The allocation is that many updates "
                  f"behind what the cadence says it is.")
        if first.get("alloc_clip_hits", 0):
            print(f"  !! the allocation hit its |a| clip "
                  f"{first['alloc_clip_hits']} times: a group wanted more than "
                  f"anchor_alloc_clip_decades of separation and did not get it.")


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

    print(f"{stem} written to {jsonl_path} and {csv_path} "
          f"({len(history)} rows)")
    return [jsonl_path, csv_path]

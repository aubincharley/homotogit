"""Generate and push one Kaggle kernel per method, for one dataset.

One kernel per method rather than one for all four: each then fits comfortably
inside the session limit, they queue independently, and a failure costs one run
rather than the campaign.

The kernel clones this repository at a pinned commit instead of carrying the
code as a dataset, so ``Trainer.provenance()`` records a real git SHA in every
``summary.json``.  That needs ``enable_internet``; if internet has to stay off,
push the tree as a dataset, attach it, and drop the clone step -- at the cost of
``code_version()`` returning ``None``.

Usage::

    py scripts/kaggle/push_runs.py --dataset svhn --dry-run   # write, push nothing
    py scripts/kaggle/push_runs.py --dataset svhn --seeds 0

Then, once a kernel reports ``complete``::

    kaggle kernels status  aubincharley/<dataset>-<method>-seed0
    kaggle kernels output  aubincharley/<dataset>-<method>-seed0 -p runs/<dataset>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

REPO = "https://github.com/aubincharley/homotogit.git"
BRANCH = "activation-transfer"

#: ``slug`` is the kernel-name prefix, defaulting to the key.  Kaggle rejects a
#: kernel slug longer than ~50 characters with a bare 400 and no explanation, and
#: "<key>-resolution-max-b1-gaussian-conv-seed0" is 59 for the arm names below.
#:
#: Per dataset: the Kaggle dataset to attach, the file the kernel globs for to
#: locate it, the config script, the asset directory, and any extra make-assets
#: arguments.  ``epochs`` sizes the permutation arrays, which must cover the
#: budget; generate with headroom, since regenerating changes the .npz digest.
DATASETS = {
    "stl10": {
        # 360 MB, labelled only.  pratt3000/stl10-binary-files has the
        # stl10_binary/ folder name already but drags in a 2.7 GB
        # unlabeled_X.bin that continuation_core.data never reads.
        "source": "yellowflag/stl10labeled2",
        "marker": "train_X.bin", "link": "data/stl10_binary",
        "configs": "scripts/stl10_configs.py", "assets": "assets/stl10_resnet20bn",
        "epochs": "90", "extra": [],
    },
    # CIFAR-10 cut to 5,000 images, two schedule arms.  Same data and assets for
    # both; only --arm differs, so they isolate the schedule.
    "cifar10_5k_stl_budget": {
        "slug": "c5k-stl",
        "source": "aubincharley/cifar-10-batches-py",
        "marker": "data_batch_1", "link": "data/cifar-10-batches-py",
        "configs": "scripts/cifar10_subset_configs.py",
        "assets": "assets/cifar10_5k_resnet20bn",
        "dataset_arg": "cifar10", "epochs": "300",
        "extra": ["--subset-size", "5000"],
        "config_extra": ["--arm", "stl_budget"],
        "config_subdir": "stl_budget",
    },
    "cifar10_5k_reference_updates": {
        "slug": "c5k-ref",
        "source": "aubincharley/cifar-10-batches-py",
        "marker": "data_batch_1", "link": "data/cifar-10-batches-py",
        "configs": "scripts/cifar10_subset_configs.py",
        "assets": "assets/cifar10_5k_resnet20bn",
        "dataset_arg": "cifar10", "epochs": "300",
        "extra": ["--subset-size", "5000"],
        "config_extra": ["--arm", "reference_updates"],
        "config_subdir": "reference_updates",
    },
    # VGG-11-BN on the unchanged CIFAR-10 reference recipe: a different
    # architecture, the same data in the same order as the ResNet-20 reference.
    "vgg11_cifar10": {
        "slug": "vgg11",
        "source": "aubincharley/cifar-10-batches-py",
        "marker": "data_batch_1", "link": "data/cifar-10-batches-py",
        "configs": "scripts/vgg11_configs.py", "assets": "assets/vgg11_cifar10",
        "dataset_arg": "cifar10", "arch": "vgg11_bn", "epochs": "30",
        # the initial weights must be new (different parameter shapes, so
        # --init-from correctly refuses), but the data order is the reference's
        "extra": ["--indices-from", "assets/cifar10_resnet20bn"],
        "no_init_from": True,
    },
    # ResNet-20 with GroupNorm: the BatchNorm control. Same recipe, same data in
    # the reference's own order, same sites; only the normalisation differs.
    "resnet20gn_cifar10": {
        "slug": "r20gn",
        "source": "aubincharley/cifar-10-batches-py",
        "marker": "data_batch_1", "link": "data/cifar-10-batches-py",
        "configs": "scripts/resnet20_gn_configs.py",
        "assets": "assets/resnet20gn_cifar10",
        "dataset_arg": "cifar10", "arch": "resnet20_gn_cifar", "epochs": "30",
        # GroupNorm has no running buffers, so a BatchNorm state dict will not
        # load and --init-from correctly refuses; the data order is the reference's
        "extra": ["--indices-from", "assets/cifar10_resnet20bn"],
        "no_init_from": True,
    },
    # ResNet-20-BN with GELU / SiLU: the ReLU control. No asset set of its own --
    # the state dict is interchangeable with resnet20_bn, so these load the pinned
    # reference assets directly, same initial weights and same data order.
    "r20gelu_cifar10": {
        "slug": "r20gelu",
        "source": "aubincharley/cifar-10-batches-py",
        "marker": "data_batch_1", "link": "data/cifar-10-batches-py",
        "configs": "scripts/resnet20_act_configs.py",
        "assets": "assets/cifar10_resnet20bn",
        "dataset_arg": "cifar10", "arch": "resnet20_act_cifar", "epochs": "30",
        "extra": [], "skip_assets": True,
        "config_extra": ["--activation", "gelu"], "config_subdir": "gelu",
    },
    "r20silu_cifar10": {
        "slug": "r20silu",
        "source": "aubincharley/cifar-10-batches-py",
        "marker": "data_batch_1", "link": "data/cifar-10-batches-py",
        "configs": "scripts/resnet20_act_configs.py",
        "assets": "assets/cifar10_resnet20bn",
        "dataset_arg": "cifar10", "arch": "resnet20_act_cifar", "epochs": "30",
        "extra": [], "skip_assets": True,
        "config_extra": ["--activation", "silu"], "config_subdir": "silu",
    },
    "svhn": {
        "source": "aubincharley/svhn-binary",
        "marker": "train_X.bin", "link": "data/svhn_binary",
        "configs": "scripts/svhn_configs.py", "assets": "assets/svhn_resnet20bn",
        # 50,000 of 73,257, so updates/epoch matches the CIFAR-10 reference
        "epochs": "30", "extra": ["--subset-size", "50000"],
    },
}

KERNEL = '''"""{dataset} continuation run: {method}, seed {seed}.

Generated by scripts/kaggle/push_runs.py -- edit that, not this.
Commit {commit}.
"""
import glob
import os
import subprocess
import sys

REPO_DIR = "/tmp/homotogit"
OUT = "/kaggle/working/runs"
METHOD = "{method}"
SEED = {seed}
COMMIT = "{commit}"
DATASET = "{dataset}"
DATA_ARG = "{dataset_arg}"
CONFIG_DIR = "{config_subdir}"
LINK = "{link}"
MARKER = "{marker}"
CONFIGS = "{configs}"
ASSETS = "{assets}"
ARCH = "{arch}"


def run(*cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


# -- hardware, first, so the log says what this ran on ------------------------
import torch
if not torch.cuda.is_available():
    raise SystemExit("no GPU: the accelerator request did not take effect, and "
                     "96x96 on CPU is not worth the session")
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0),
      "| capability sm_%d%d" % torch.cuda.get_device_capability(0), flush=True)
# is_available() is not enough: a Tesla P100 (sm_60) reports True under a torch
# built for sm_70+, then fails on the first real kernel launch.  Launch one now.
try:
    torch.linalg.norm(torch.ones(64, 64, device="cuda") @ torch.ones(64, 64, device="cuda"))
    torch.cuda.synchronize()
except RuntimeError as exc:
    raise SystemExit("GPU present but unusable by this torch build (%s): %s"
                     % (torch.cuda.get_device_name(0), exc))


# -- code: a pinned commit, so summary.json provenance records a real SHA -----
run("git", "clone", "--branch", "{branch}", "--single-branch", "{repo}", REPO_DIR)
run("git", "-C", REPO_DIR, "checkout", "--detach", COMMIT)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

# -- data --------------------------------------------------------------------
# continuation_core.data wants <root>/<name>_binary/ .  Kaggle mounts a dataset
# under a path we should not have to guess (and guessed wrong once), and some of
# these datasets keep the binaries at their root rather than in that folder.  So
# find the marker file and link whatever directory holds it.
print("/kaggle/input:", sorted(os.listdir("/kaggle/input")), flush=True)
found = sorted(glob.glob("/kaggle/input/**/" + MARKER, recursive=True))
if not found:
    raise SystemExit("no %s under /kaggle/input; tree: %s"
                     % (MARKER, sorted(glob.glob("/kaggle/input/**/*", recursive=True))[:40]))
src = os.path.dirname(found[0])
os.makedirs(os.path.dirname(LINK), exist_ok=True)
if os.path.islink(LINK):
    os.unlink(LINK)
os.symlink(src, LINK)
print(LINK, "->", src, sorted(os.listdir(LINK)), flush=True)

PY = [sys.executable, "-m", "continuation_core"]

# -- assets: new index arrays, initial weights copied from the CIFAR-10 set ---
# Epochs are generated with headroom: Trainer only needs perms.shape[0] >=
# budget, and regenerating later would change the .npz digest and void pairing
# with these runs.  With --init-from no torch RNG is touched, so this is
# bit-identical to the set any other machine would produce.
if {skip_assets}:
    # this study reuses a committed asset set as-is, so there is nothing to mint;
    # verify it instead, which is the same check make-assets would have left behind
    print("reusing committed assets:", ASSETS, flush=True)
    run(*PY, "verify-assets", "--assets", ASSETS)
else:
    run(*PY, "make-assets", "--dataset", DATA_ARG, "--data-root", "data",
        "--arch", ARCH, "--epochs", "{epochs}", "--seeds", "0,1,2",
        "--out", ASSETS, {init_from}{extra})

# -- configs: every transfer decision lives in the dataset's config script ----
run(sys.executable, CONFIGS, "--data-root", "data",
    "--assets", ASSETS, "--seeds", str(SEED),
    "--out", "configs/" + DATASET, "--run-out", OUT, {config_extra})

CONFIG = os.path.join("configs", DATASET, CONFIG_DIR, "%s__seed%d.json" % (METHOD, SEED))
print(open(CONFIG).read(), flush=True)

# -- dry run: aborts the kernel unless the target state is bitwise plain ------
run(*PY, "dry-run", "--config", CONFIG, "--report", OUT + "/dryrun_%s.json" % METHOD)

# -- train -------------------------------------------------------------------
run(*PY, "train", "--config", CONFIG, "--out", OUT)
run("cp", "-r", "configs/" + DATASET, OUT + "/configs")
run("cp", ASSETS + "/assets_manifest.json", OUT + "/%s_assets_manifest.json" % DATASET)
print("done", flush=True)
'''

METADATA = {
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    # enable_gpu alone gave a Tesla P100 (sm_60), which Kaggle's torch
    # 2.10.0+cu128 does not support (it needs sm_70+): cuda.is_available() is
    # True but every kernel launch fails.  The reference CIFAR-10 runs were on
    # T4s, so ask for the same card by name.
    "machine_shape": "NvidiaTeslaT4",
    "enable_tpu": False,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
}


#: Kaggle allows two concurrent batch GPU sessions and rejects the next push
#: outright.  Counting only the sessions *this* process started is not enough --
#: anything already running, from an earlier invocation or another terminal,
#: occupies a slot too -- so retry on the error Kaggle actually returns.
BUSY = "Maximum batch GPU session count"
POLL_SECONDS = 60
MAX_WAIT_SECONDS = 3 * 60 * 60


def push(folder: Path) -> bool:
    """Push, waiting out a full GPU queue rather than dropping the kernel.

    Returns False on a real failure instead of raising: one bad kernel must not
    abandon the twenty after it.
    """
    deadline = time.time() + MAX_WAIT_SECONDS
    while True:
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(folder)],
                           capture_output=True, text=True)
        out = r.stdout + r.stderr
        if BUSY not in out:
            if r.returncode != 0 or "Error" in out:
                print("  FAILED %s: %s" % (folder.name, " | ".join(
                    l.strip() for l in out.splitlines() if l.strip())[:300]), flush=True)
                return False
            print("  pushed", folder.name, flush=True)
            return True
        if time.time() > deadline:
            print("  FAILED %s: no GPU slot after %d s" % (folder.name, MAX_WAIT_SECONDS),
                  flush=True)
            return False
        print("  GPU queue full; retrying %s in %ds" % (folder.name, POLL_SECONDS),
              flush=True)
        time.sleep(POLL_SECONDS)


def already_complete(user: str, slug: str) -> bool:
    """Has this kernel already run to completion?

    A push re-runs the kernel, so re-pushing a finished one burns GPU quota for a
    result already in hand.  That matters when a pusher dies part way through a
    campaign, which is what this guard is for.
    """
    r = subprocess.run(["kaggle", "kernels", "status", "%s/%s" % (user, slug)],
                       capture_output=True, text=True)
    return "COMPLETE" in r.stdout


def head_commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def commit_is_on_origin(commit: str) -> bool:
    r = subprocess.run(["git", "-C", str(ROOT), "branch", "-r", "--contains", commit],
                       capture_output=True, text=True)
    return r.returncode == 0 and "origin/" in r.stdout


def write(dataset: str, method: str, seed: int, commit: str, user: str,
          out_root: Path) -> Path:
    spec = DATASETS[dataset]
    slug = "%s-%s-seed%d" % (spec.get("slug", dataset),
                             method.replace("_", "-"), seed)
    if len(slug) > 50:
        raise SystemExit("kernel slug %r is %d characters; Kaggle rejects over 50 "
                         "with a bare 400. Shorten the dataset's 'slug' field."
                         % (slug, len(slug)))
    d = out_root / slug
    d.mkdir(parents=True, exist_ok=True)
    extra = ", ".join('"%s"' % a for a in spec["extra"])
    config_extra = ", ".join('"%s"' % a for a in spec.get("config_extra", []))
    (d / (slug + ".py")).write_text(KERNEL.format(
        dataset=dataset, method=method, seed=seed, commit=commit, repo=REPO,
        branch=BRANCH, link=spec["link"], marker=spec["marker"],
        configs=spec["configs"], assets=spec["assets"], epochs=spec["epochs"],
        arch=spec.get("arch", "resnet20_bn_cifar"),
        skip_assets=bool(spec.get("skip_assets")),
        init_from=("" if spec.get("no_init_from")
                   else '"--init-from", "assets/cifar10_resnet20bn", '),
        extra=extra, config_extra=config_extra,
        dataset_arg=spec.get("dataset_arg", dataset),
        config_subdir=spec.get("config_subdir", "")))
    meta = dict(METADATA, id="%s/%s" % (user, slug),
                # the title must slugify to the id, or Kaggle warns and the
                # two can drift apart
                title=slug.replace("-", " "),
                dataset_sources=[spec["source"]],
                code_file=slug + ".py")
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--methods",
                    default="plain,resolution_max_b1,gaussian_postrelu,"
                            "resolution_max_b1_gaussian_conv")
    ap.add_argument("--user", default="aubincharley")
    ap.add_argument("--commit", help="default: HEAD, which must be pushed to origin")
    ap.add_argument("--out", default=str(Path(__file__).parent / "kernels"))
    ap.add_argument("--dry-run", action="store_true", help="write the kernels, push nothing")
    ap.add_argument("--skip-complete", action="store_true",
                    help="do not re-push kernels that already ran to completion; "
                         "use when resuming a campaign whose pusher died")
    args = ap.parse_args(argv)

    commit = args.commit or head_commit()
    if not commit_is_on_origin(commit):
        print("warning: %s is not on any origin branch; the kernel's git clone will "
              "fail until you push it" % commit[:12], file=sys.stderr)

    pushed, failed, skipped = [], [], []
    for seed in (int(s) for s in args.seeds.split(",")):
        for method in args.methods.split(","):
            d = write(args.dataset, method, seed, commit, args.user, Path(args.out))
            print("wrote", d)
            if args.dry_run:
                continue
            if args.skip_complete and already_complete(args.user, d.name):
                print("  already complete, skipping", d.name, flush=True)
                skipped.append(d.name)
                continue
            (pushed if push(d) else failed).append(d.name)
    print("\npushed %d kernel(s)%s"
          % (len(pushed), ", skipped %d already complete" % len(skipped) if skipped else ""))
    if failed:
        print("FAILED %d: %s" % (len(failed), ", ".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

# Running on Kaggle

Training happens on Kaggle; this machine only runs the tests and `dry-run`.

```bash
py scripts/kaggle_run.py <job.py> --gpu
```

* The job must live inside the repo and write to `/kaggle/working/` — that is
  what comes back.
* Results land in `results/kaggle_outputs/<slug>/`, with a `launch.json`
  recording the account, kernel id, datasets and UTC launch time. A non-empty
  output directory is refused unless `--overwrite`.
* Blocking: polls every 20 s, default timeout 1 h. Raise it with
  `--timeout-seconds` (28800 = 8 h is what a 12-cell batch needs).
* Internet is off unless `--internet`. There is no remote shell — a fresh
  container runs one script.
* GPU quota is roughly 30 h/week, so don't spend it on CPU-only work.
  `--dry-run` packages and reports without pushing.

## Authentication

Either an API token (`~/.kaggle/kaggle.json`) or a browser login
(`~/.kaggle/credentials.json`, written by `kaggle auth login`). `kaggle_run.py`
reads the username from `KAGGLE_USERNAME`, then the token, then the OAuth
credentials — the last of which has no `key` field, which is why the plain
`json["username"]` lookup is not enough.

`--account <name>` switches to `~/.kaggle-accounts/<name>/kaggle.json` by
exporting `KAGGLE_CONFIG_DIR` before anything authenticates. Quotas and kernel
ownership follow the selected account. `kernels list --mine` only sees the
selected account, so check before relaunching to avoid submitting twice.

## How the code gets there

A Kaggle script kernel accepts one file, so `kaggle_run.py` zips the requested
packages, base64-encodes them into the generated `script.py`, and unpacks them
to `/kaggle/working/_repo` on the worker before running the job. The kernel
therefore runs the code that is checked out here, not a separately maintained
copy.

The source cap is 900 kB. `continuation_core` plus `scripts` packages to about
78 kB, so it fits with room to spare.

Anything bigger goes as a mounted dataset — which is why the 15 MB of pinned
assets are published separately (`scripts/stage_assets.py`) rather than shipped
in the source.

## Mounted inputs

Kaggle inserts a path segment of its own under `/kaggle/input` (observed:
`/kaggle/input/datasets/<owner>/<slug>/`), so jobs glob for a known filename
instead of assuming the layout:

| input | how it is found | override |
|---|---|---|
| CIFAR-10 | `/kaggle/input/**/cifar-10-batches-py` | `STUDY_DATA` |
| pinned assets | `/kaggle/input/**/assets_manifest.json` | `CORE_ASSETS` |

Datasets in use: `alexandrecorrard/cifar-10-batches-py` for the data, and the asset set
published by `scripts/stage_assets.py`.

## What a job owes you

The wrapper always writes `environment.json` (python, platform, torch, CUDA,
whether a GPU was **actually** granted, its name and VRAM). An accelerator is
requested, never assumed: `job_optimizer_benchmark.py` writes `ABORTED.json`
and exits rather than spend a session on CPU.

Pinned assets are verified on the worker before any training starts. That check
is what ties a run to the recorded results it will be compared against; without
it a comparison is not meaningful, so it fails the job rather than warning.

Sessions are capped. Every run writes `rolling.pt` each epoch, and a relaunched
job resumes each unfinished cell from it — mid-epoch, with the RNG — so a cut
session costs minutes, not a run.

## Splitting a campaign

`job_optimizer_benchmark.py` builds its cell list once, sorts it
deterministically and takes `index % OPT_BENCH_NJOBS == OPT_BENCH_JOB`. A cell
belongs to exactly one kernel however the slices are launched or relaunched.
Within a kernel, cells are dealt longest-first across the granted GPUs, because
the methods differ by up to 45 % in cost.

`OPT_BENCH_BATCH`, `OPT_BENCH_JOB` and `OPT_BENCH_NJOBS` select the batch and
the slice. `kaggle_run.py` pushes one script and passes it no environment, so
each slice is its own small entry point that sets those variables and then runs
the job:

| launcher | cells | measured estimate |
|---|---:|---:|
| `job_optbench_lr_sweep.py` | 12 | ~50 min |
| `job_optbench_sgd_control.py` | 3 | ~13 min |
| `job_optbench_grid_0.py` … `_2.py` | 12 each | ~60 min each |

```bash
py scripts/kaggle_run.py scripts/job_optbench_lr_sweep.py --gpu \
    --accelerator NvidiaTeslaT4 \
    --dataset alexandrecorrard/cifar-10-batches-py \
    --dataset alexandrecorrard/continuation-core-r20bn-assets \
    --include continuation_core --include scripts \
    --timeout-seconds 28800
```

Order matters: the sweep has to be read out into `docs/LR_SWEEP.md` and its
result written into `CHOSEN_LR` in `job_optimizer_benchmark.py` before the grid
launchers will run at all — they exit with an error rather than pick a learning
rate for you.

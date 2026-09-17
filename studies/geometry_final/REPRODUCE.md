# Reproducing the geometry completion

## Code and versions

- **Branch:** `geometry-final`, started from `visualization@9e9275c`. `geometry_final/` is the only addition.
- **Unchanged modules:** `landscape_v3.{common,evaluator,hessian,run}`, `landscape_v2`, `landscape_study` and `continuation_core`. `git diff 9e9275c` on them is empty.
- **Kaggle (T4):** torch 2.10.0+cu128, CUDA 12.8, numpy 2.0.2, Python 3.12, 2 × Tesla T4, TF32 off, cuDNN deterministic.
- **Local analysis:** Python 3.12, torch 2.5.1+cu121, numpy 1.26.3, pandas 2.2.0, matplotlib 3.8.2.

## Inputs (not shipped)

| input | location | identity |
|---|---|---|
| 20 final checkpoints `<method>__seed<k>__epoch_030.pt` | Kaggle dataset `maxmonstre/landscape-v3-inputs` (private); originals in the `visualization` worktree, `studies/landscape_v2/raw/<account>/v2/runs/<run>/checkpoints/epoch_030.pt` | sha256 in `protocol/input_hashes.json` |
| direction draws `seed0_dir00.pt`, `seed0_dir01.pt` | same dataset; `studies/landscape_v2/raw/maxnicaise/v2/directions/` | sha256 in `protocol/input_hashes.json` |
| `subsets.npz` (probes, calib2k) | same dataset; `studies/landscape_v2/inputs/subsets.npz` | sha256 45ec962b… |
| CIFAR-10 python batches | Kaggle `pankrzysiu/cifar10-python` | official batches |
| existing landscape_v3 raw outputs and tables | `visualization` worktree, `studies/landscape_v3/{raw,tables}` (uncommitted) | per-job `eval_manifest.json` sha256 |

## Commands

```bash
py -m geometry_final.plan
```

Rebuilds `protocol/tasks.json` from the existing raw outputs. Its sha256 is recorded in `raw/launch_*.json`.

```bash
py -m geometry_final.kaggle push --tag pilot
```

```bash
py -m geometry_final.kaggle push --tag main
```

```bash
py -m geometry_final.kaggle pull --tag main
```

```bash
py -m geometry_final.kaggle verify --tag main
```

`verify` recomputes the sha256 of every file listed in `manifest.json`.

```bash
py -c "from geometry_final.trace import dense_reference; print(dense_reference())"
```

Dense-reference check of the trace estimator.

```bash
py -m geometry_final.existing
```

Analyses of existing measurements (X1–X5 tables).

```bash
py -m geometry_final.traces
```

Trace tables (T_trace_*), control variate, paired comparisons.

```bash
py -m geometry_final.figures
```

```bash
py -m geometry_final.curvature_figs
```

```bash
py -m geometry_final.handoff
```

LaTeX tables and macros.

```bash
py -m geometry_final.package --dest <folder>
```

## Environment variable

`GEOMETRY_V3_ROOT` points at the worktree holding `studies/landscape_v3` and `studies/landscape_v2`. The default is the sibling `visualization` worktree.

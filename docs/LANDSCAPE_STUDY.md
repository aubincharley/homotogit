# Loss-landscape study v1: commands

Evaluation only. No training is launched anywhere in this study. Package:
`landscape_study/`. Inputs, results, figures and report:
`studies/landscape_v1/`.

## 1. Prepare (local, CPU, once)

```bash
py -m landscape_study.prepare --data-root ../Projet_filiere/data
```

Refuses to run if `inputs/preregistration.json` already exists. It performs these steps:

- **Provenance:** checks the resbench runs (asset digests, recipe, recorded per-epoch states) and writes `sources_report.json`.
- **Subsets:** draws the class-balanced probes and the calibration subset.
- **Directions:** draws the underlying tensors for direction pairs 0–4.
- **Checkpoints:** strips the 24 checkpoints and the 3 initial states into `inputs/weights/`.
- **PCA:** computes the shared PCA plane.
- **Records:** writes the preregistration and the sha256 manifest.

The weights folder is git-ignored. It can be rebuilt from the resbench outputs, and the manifest pins its content.

## 2. Local pilot and range check

```bash
py -m landscape_study.run --job pilot --inputs studies/landscape_v1/inputs --data-root ../Projet_filiere/data --out pilot_out
```

```bash
py -m landscape_study.run --job offsets --inputs studies/landscape_v1/inputs --data-root ../Projet_filiere/data --out studies/landscape_v1/results/offsets --skip-checks
```

Every job first runs `landscape_study/checks.py` and writes `checks_job<k>.json`.

## 3. Full evaluation on three Kaggle accounts

```bash
py -m landscape_study.kaggle publish --accounts maxnicaise maxlefrr maxnikezz
```

```bash
py -m landscape_study.kaggle push --job 0 --account maxnicaise
```

```bash
py -m landscape_study.kaggle push --job 1 --account maxlefrr
```

```bash
py -m landscape_study.kaggle push --job 2 --account maxnikezz
```

```bash
py -m landscape_study.kaggle pull --job 0 --account maxnicaise
```

| job | content |
|---|---|
| 0 | A: 21x21 grids, plain and resolution-only, epoch 30, seed 0, pair 0 |
| 1 | B: 21x21 grids, resolution-only epoch 6 under r = 16 / 24 / 32; 1-D slices: epoch 12, and plain epoch-6 weights with the hook |
| 2 | A robustness 1-D slices (3 seeds, 6 directions each); C checkpoint and plane evaluations; D interpolations (3 seeds); saved-statistics vs recalibrated for all 24 checkpoints |

The same jobs run locally with `--job 0|1|2` or `--job all`. Results are
resumable: rerunning with the same `--out` skips completed points.

## 4. Figures and summary

```bash
py -m landscape_study.figures --results studies/landscape_v1/results
```

Writes PDF and PNG files plus `summary.json` to `studies/landscape_v1/figures/`.

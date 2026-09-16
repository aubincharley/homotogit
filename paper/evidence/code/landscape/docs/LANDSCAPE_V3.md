# landscape_v3: amplitude, validation, temporal and Hessian extension

This extension uses only the 20 `landscape_v2` networks (4 methods × seeds 0–4)
and their saved checkpoints. There is no training.

- **Protocol:**
  [studies/landscape_v3/protocol/PROTOCOL.md](../studies/landscape_v3/protocol/PROTOCOL.md)
- **Report:** [studies/landscape_v3/REPORT.md](../studies/landscape_v3/REPORT.md)
- **Figure index:** [studies/landscape_v3/FIGURES.md](../studies/landscape_v3/FIGURES.md)

## Commands

```bash
py -m landscape_v3.pilot dense
```

```bash
py -m landscape_v3.plan verify
```

```bash
py -m landscape_v3.plan matrix
```

```bash
py -m landscape_v3.kaggle publish
```

```bash
py -m landscape_v3.kaggle push --pilot --accounts maxfrrsava
```

```bash
py -m landscape_v3.kaggle push
```

```bash
py -m landscape_v3.kaggle status
```

```bash
py -m landscape_v3.kaggle pull --account maxnicaise
```

```bash
py -m landscape_v3.kaggle verify --account maxnicaise
```

```bash
py -m landscape_v3.analyze
```

What each command does:

- `pilot` runs the local numerical pilots: `dense`, `real`, `lanczos` and
  `evalcost`.
- `plan verify` checks the checkpoints, manifests, states, digests and pairing.
- `plan matrix` builds the evaluation matrix, subtracts reusable v2 points, and
  writes the per-account bundles.
- `kaggle publish` creates the per-account input datasets.
- `kaggle push --pilot` runs the T4 timing and numerics pilot.
- `kaggle push` launches the main jobs.
- `kaggle verify` recomputes the sha256 of every file in the job manifest after
  a download.
- `analyze` produces `tables/`, `figures/` and `tables/summary.json`.

## Modules

| file | role |
|---|---|
| `common.py` | frozen constants, canonical point keys (`pkey`), account layout |
| `evaluator.py` | three BatchNorm policies (`saved`, `recalibrated` = pointwise, `centre_frozen`), float32/float64 |
| `hessian.py` | frozen-BN objective, exact HVPs, relative coordinates, r(δ), Lanczos with full reorthogonalisation |
| `v2points.py` | maps v2 raw evaluations to canonical keys (reuse) |
| `plan.py` | input verification, evaluation matrix, per-account task bundles, reuse samples |
| `run.py` | Kaggle job: input digests → reuse reproduction → GPU task queue (points, grids, hchecks, Hessian, quadratic forms, cuts, planes) → manifest |
| `kaggle.py` | publish / push / status / pull / verify |
| `pilot.py` | local numerical and timing pilot |
| `acommon.py`, `asens.py`, `ahess.py`, `asurf.py`, `analyze.py` | analysis |

## Output layout

```
studies/landscape_v3/
  protocol/   PROTOCOL.md, numerical_protocol.json, input_verification.json,
              expected_digests.json, evaluation_matrix.json, bundle_<account>.json
  pilot/      local pilot outputs
  raw/<account>/v3/   environment.json, timing.json, eval_manifest.json,
                      checks/, eval/<task>.jsonl + .meta.json,
                      hessian/<problem>.json + _vectors.pt
  tables/, figures/, REPORT.md, FIGURES.md
```

## Reading a point

A point key is:

`method|s<seed>|<checkpoint>|r<res>_G<level>|<policy>|<f32/f64>|<probes/large>|<spec>`

where `<spec>` is one of:

| spec | meaning |
|---|---|
| `c` | the centre |
| `r1:k:ε` | random direction k, signed amplitude ε |
| `r2:0:1:a:b` | random plane |
| `h1:<problem>:<vector>:t` | Hessian cut |
| `h2:<problem>:<v1>:<v2>:u1:u2` | Hessian plane, unit grid |

The analysis joins v3 rows and reused v2 rows on this key.

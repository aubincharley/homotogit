# Loss-landscape study v2: commands

Package: `landscape_v2/`. Study directory: `studies/landscape_v2/`.
`landscape_study/` and `studies/landscape_v1/` (v1) are untouched.

## 1. Inputs (local, once)

```bash
py -m landscape_v2.assets_v2
```

```bash
py -m landscape_v2.prepare
```

- **`assets_v2`** writes five seeds. Seeds 0–2 are the pinned reference
  assets, re-checked by digest; seeds 3–4 are generated reproducibly.
- **`prepare`** freezes the subsets, directions rule, amplitudes, transitions,
  primary figure choices and priorities in `inputs/preregistration.json`, with
  a sha256 manifest. It refuses to overwrite an existing preregistration.

## 2. Local pilot

The local pilot has two parts:

- **Training check:** a short training run of each method with a resume,
  checking the checkpoint format and transition windows.
- **Evaluation check:** the same run loop on those checkpoints in pilot mode,
  reduced to tiny grids with `--pilot`:

```bash
py -m landscape_v2.run --stage eval --seeds 3 --pilot --inputs studies/landscape_v2/inputs --data-root ../Projet_filiere/data --out PILOT_OUT
```

## 3. Kaggle: one private T4×2 kernel per account

```bash
py -m landscape_v2.kaggle publish
```

```bash
py -m landscape_v2.kaggle push
```

```bash
py -m landscape_v2.kaggle status
```

```bash
py -m landscape_v2.kaggle pull
```

```bash
py -m landscape_v2.kaggle verify
```

Seeds per account are set in `common.ACCOUNT_SEEDS`. Each kernel:

1. trains its seeds (four methods per seed, two GPU workers, longest method
   first);
2. writes `runs_manifest.json`;
3. evaluates in priority order, stopping new points at the deadline;
4. writes `eval_manifest.json` and `timing.json`.

`verify` recomputes every sha256 of the downloaded files against those
manifests.

## 4. Analysis

```bash
py -m landscape_v2.analyze
```

It writes figures to `studies/landscape_v2/figures/` and CSV files plus
`summary.json` to `studies/landscape_v2/tables/`.

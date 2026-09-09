# Kaggle compute — agent quickstart

To run something on Kaggle instead of locally:

```
py scripts/kaggle_run.py <your_script.py> [--gpu]
```

- Script must write outputs to `/kaggle/working/` — that's what comes back.
- Results land in `results/kaggle_outputs/<slug>/` (script filename = slug, unless `--title` given).
- Blocks until done (polls every 20s, 1h default timeout — use `--timeout-seconds` for longer jobs).
- Add `--gpu` for GPU. Quota is limited (~30h/week), don't use it for CPU-only work.
- No shell access remotely — only the one script you push runs, in a fresh container each time.

Auth is already set up (`~/.kaggle/kaggle.json`). Nothing else to configure.

## Multiple accounts

Credentials live one directory per account:

```
~/.kaggle-accounts/<name>/kaggle.json     # chmod 600
~/.kaggle/kaggle.json                     # still the default
```

`scripts/kaggle_run.py --account <name>` exports `KAGGLE_CONFIG_DIR` for the CLI
subprocesses before anything authenticates. Without the flag the default
`~/.kaggle` is used, so existing commands are unchanged. The kernel owner, the
kernel id and the quota all follow the selected account, and every launch writes
`results/kaggle_outputs/<slug>/launch.json` recording which account ran it.

Configured accounts: `maxnicaise`, `maxlefrr`. Both were measured on the same
image — 2x Tesla T4, `torch 2.10.0+cu128` — which is why digests reproduce across
them.

`kernels list --mine` only sees the selected account's kernels, so check both
before launching to avoid duplicate submissions.

### Pairing across accounts

Comparisons are only valid between runs with the same initial weights, BN
buffers, subset, probe and batch order. `scripts/aggregate_runs.py` hashes those
artifacts for every downloaded run and groups runs by *pairing key*; runs in one
group may be differenced, runs in different groups must be labelled contextual.
`require_paired()` raises rather than returning a quietly wrong number.

Keep any set of arms that must be compared *within one job*. Use the second
account for a different seed or a different experiment running in parallel.

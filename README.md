# Continuation methods for CNN training: exploratory benchmark (organized)

This branch, `benchmark-organized`, is the complete record of the exploratory
phase on CIFAR-10: every operator, schedule, launcher, raw result and figure,
plus a machine-readable index of all experiments. The minimal code for the next
phase is on branch **`continuation-core`**.

## Where things are

| looking for | go to |
|---|---|
| protocol, vocabulary (current vs target path, pairing, schedules) | [docs/PROTOCOL.md](docs/PROTOCOL.md) |
| findings, limitations, abandoned approaches | [docs/FINDINGS.md](docs/FINDINGS.md) |
| every trained configuration with its conditions | [docs/BENCHMARK_TABLE.md](docs/BENCHMARK_TABLE.md) (generated) |
| machine-readable index: IDs, commits, asset sets, seeds, digests, paths, validity, timing, file locations | [experiments/index.json](experiments/index.json), schema in [experiments/README.md](experiments/README.md) |
| corrections to earlier tables and claims | [docs/AUDIT.md](docs/AUDIT.md) |
| full history with exact numbers | [docs/HANDOVER.md](docs/HANDOVER.md) |
| operators and schedules | `continuation/` (`campaign_ops.py`, `ablation_ops.py`, `resolution_ops.py`, `adaptive.py`, `transforms/`), docs in [docs/README.md](docs/README.md) |
| configurations and launchers | `scripts/*_manifest.py`, `scripts/job_*.py`, drivers; map in [scripts/README.md](scripts/README.md) |
| raw results | `results/kaggle_outputs/<kernel-slug>/` (summaries, metrics, manifests; checkpoints not versioned). Idriss's ablation and Aubin's sigma0 sweep stay on their branches, located by commit in the index |
| aggregation and figures | `scripts/build_experiment_index.py`, `scripts/benchmark_table.py`, `scripts/gather_all_methods.py`, `scripts/plot_*.py`, `results/presentation/` |
| knowledge base (French) | [docs/research/continuation/](docs/research/continuation/README.md) |
| archived entry points | [archive/README.md](archive/README.md) |

## Rebuild the index and the table

```bash
git fetch origin
py scripts/build_experiment_index.py
py scripts/benchmark_table.py
py scripts/plot_all_methods.py
```

Reading records from teammates' branches needs their commits locally (hence
`git fetch`). To check the shipped Kaggle code bundles, set `BUNDLE_ROOT` to the
checkout that downloaded the outputs; those bundles are git-ignored.

## Install and test

```bash
py -m pip install -r requirements.txt
py -m pytest -q
```

Two execution paths remain, both historical: `continuation.cli` with YAML
configs for the input-space experiments, and `scripts/*_driver.py` with
`scripts/kaggle_run.py` for the feature-space and resolution campaigns (see
[docs/kaggle_cli.md](docs/kaggle_cli.md)).

## Status

No campaign is running or planned on this branch. The three methods chosen for
the next phase, their exact definitions, and the parity checks against this
code are on `continuation-core` (`docs/METHODS.md`,
`verification/parity_report.json`).

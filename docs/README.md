# Documentation map

| I want… | read |
|---|---|
| the protocol and the vocabulary (current vs target path, pairing, schedules) | [PROTOCOL.md](PROTOCOL.md) |
| what was found, its limits, and what was abandoned | [FINDINGS.md](FINDINGS.md) |
| every trained configuration with its conditions | [BENCHMARK_TABLE.md](BENCHMARK_TABLE.md) (generated) |
| the machine-readable record of all experiments | [../experiments/README.md](../experiments/README.md), `../experiments/index.json` |
| corrections made to earlier tables, labels and claims | [AUDIT.md](AUDIT.md) |
| the full chronological history with exact numbers | [HANDOVER.md](HANDOVER.md) |
| operator definitions | [gaussian.md](gaussian.md), [wavelet_shrinkage.md](wavelet_shrinkage.md), [tv_budget.md](tv_budget.md), [extensions.md](extensions.md), `continuation/campaign_ops.py`, `continuation/ablation_ops.py`, `continuation/resolution_ops.py` (docstrings) |
| Aubin's adaptive schedules | [adaptive_continuation_maths.md](adaptive_continuation_maths.md), [experiments_2026-09-10.md](experiments_2026-09-10.md) |
| the input-space experiments | [experiment0.md](experiment0.md), [exp1_warmstart.md](exp1_warmstart.md), [results.md](results.md) |
| running on Kaggle | [kaggle_cli.md](kaggle_cli.md) |
| the maintained knowledge base (French) | [research/continuation/README.md](research/continuation/README.md) |
| scripts by role | [../scripts/README.md](../scripts/README.md) |
| archived entry points | [../archive/README.md](../archive/README.md) |

The minimal code for the next phase (three frozen methods, plain control,
training, checkpointing and analysis tools) is on branch `continuation-core`.

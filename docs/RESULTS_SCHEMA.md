# Results schema `continuation_core.results/1`

One directory per run (`runs/<method>__seed<k>/` by default):

```
config.json                 the full ExperimentConfig (JSON)
environment.json            python, torch, CUDA, device, cuDNN flags, numpy
assets_verification.json    every recomputed digest of the pinned assets
metrics.json                list of evaluation records (below)
summary.json                final record (below); written only when the budget completes
rolling.pt                  latest checkpoint (overwritten each epoch)
checkpoints/epoch_NNN.pt    after epoch NNN (NNN = epochs completed)
checkpoints/update_NNNNNN.pt  transition-window / every-N-updates checkpoints
```

A real tree produced on synthetic data is in
[examples/synthetic_run/](examples/synthetic_run/). Its numbers are meaningless;
it only shows the format.

## `metrics.json` records

```jsonc
{
  "epoch": 6,                         // epochs completed; 0 = before any update
  "update": 2346,                     // optimizer updates completed
  "lr_last_update": 0.0048...,        // learning rate of the last completed update
  "state_used": {                     // null at epoch 0
    "state": {"resolution": 16, "sigma": 0.85, "label": "epoch 5"},
    "per_site_sigma": [0.425, ...]    // effective sigma at each Gaussian site
  },
  "state_next": {"state": {...}, "per_site_sigma": [...]},   // null after the last epoch
  "train_loss_epoch": 1.23,           // mean microbatch CE over the epoch, weighted by size
  "bn_policy": "running_stats",
  "eval": {
    "current": {"train_probe": {"ce": ..., "acc": ..., "n": 500},
                "test":        {"ce": ..., "acc": ..., "n": 10000}},
    "target":  {"train_probe": {...}, "test": {...}}
  },
  "elapsed_seconds": 812.4
}
```

`current` evaluates the state the epoch just used. `target` evaluates full
resolution with the Gaussian off. The two are identical once the schedule has
reached the target (epoch ≥ 21 for the reference methods).

## `summary.json`

| key | content |
|---|---|
| `schema` | `continuation_core.results/1` |
| `method`, `seed`, `dataset`, `arch`, `arch_validated` | identity |
| `validation_status` | `reference` for the unchanged preset; `modified`, `unvalidated`, … otherwise |
| `epochs`, `updates`, `updates_per_epoch`, `n_train`, `n_test` | budget and sizes |
| `final` | the last record's `eval` block (current and target) |
| `final_state_used`, `final_state_is_target` | whether the reported model is the plain network |
| `transition_updates` | first update of every new intervention state |
| `timing` | `wall_seconds`, `eval_seconds`, `train_seconds` and their scope |
| `method_definition` | full `MethodSpec` |
| `provenance` | asset digests, torch version, git commit of the code |

## Loss and analysis outputs

Every `evaluate` / `CheckpointEvaluator.loss` result carries: `split`, `n`,
`ce`, `acc`, `path` (label of the state: `used`, `next`, `target`,
`epoch k`, or a custom string), `state`, `per_site_sigma`, `bn_policy`,
`batch_size`, and for checkpoints `checkpoint`, `global_update` and
`weights` (`checkpoint` or `override`). `plane-loss` returns one surface per
(state, BN policy). `perturb` returns one block per (state, BN policy) with the
per-direction deltas. `pca-plane` writes the plane (`.pt`) plus a JSON with the
explained-variance ratios and per-point projection residuals.

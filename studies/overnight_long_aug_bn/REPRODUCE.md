# Reproduce the follow-up

Branch `overnight-long-aug-bn`, based on `comparison-cbs-sdpoint@6b28489`. Python 3.12, and torch 2.10.0+cu128 on Kaggle (2×T4 per kernel). numpy 2.0.2 on Kaggle; the local checks used torch 2.5.1 and numpy 1.26.3.

```bash
# 0. checks (CPU; the 4 Adam tests of tests/test_optimizers.py fail only on torch < 2.6, which lacks decoupled_weight_decay)
py -m pytest -q tests/test_overnight.py tests/test_comparators.py tests/test_operators_and_methods.py tests/test_training_checkpoints_analysis.py

# 1. freeze (writes protocol/, configs/, schedules.json, overnight/frozen_hashes.json)
py -m overnight.freeze

# 2. Kaggle inputs (assets; the evaluation account also gets the 42 previous final checkpoints)
py -m overnight.kaggle publish --accounts <accounts> --old-endpoints maxmonstre

# 3. pilot (tests on GPU, timing/memory of all 21 regime x arm cells, GPU resume check; no test image evaluated)
py -m overnight.kaggle push --tag pilot --mode pilot --accounts maxlebossdu91 --hours 2

# 4. P0-P3 on the previous 42 endpoints
py -m overnight.kaggle push --tag eval --mode evaluate_old --accounts maxmonstre

# 5. allocation from pilot costs, then training kernels (one run per T4, priority order per queue)
py -m overnight.allocate --pilot studies/overnight_long_aug_bn/raw/pilot/maxlebossdu91/ovn --accounts maximemonstrenikez:2:0 maxlebossdu91:2:0 maxlefrr:2:0 maxnicaise:2:0 maxnikezz:2:0 maxmonstre:2:1.0
py -m overnight.kaggle push --tag t1 --mode train --hours 10.5 --accounts <accounts>
#    a continuation after a time budget: new tag with the previous kernel output as source
py -m overnight.kaggle push --tag t2 --mode train --resume-from t1 --accounts <account>

# 6. collect, verify, analyse, plot, package
py -m overnight.kaggle pull --tag t1 --accounts <accounts>
py -m overnight.kaggle verify --tag t1 --accounts <accounts>
py -m overnight.recover_aug <extracted continuation_new_loss@badce2a> studies/overnight_long_aug_bn/recovered_augmentation
py -m overnight.analyze
py -m overnight.figures
py -m overnight.package <deliver dir>
```

A single run locally (CPU or GPU) with the frozen configuration:

```python
from overnight import matrix as MX, job
from continuation_core import assets
from continuation_core.train import Trainer
cfg = MX.config("adamw_long_aug_160", "sdpoint", 0, data_root="data", assets_dir="assets/cifar10_resnet20bn", out_dir="runs")
a = assets.load(cfg.assets.dir, 0); a["perms"] = MX.extended_perms(a["perms"], 0)
Trainer(cfg, loaded_assets=a).run()
```

Every configuration's sha256 is in `overnight/frozen_hashes.json`, and every job refuses a configuration or data order that differs from it.

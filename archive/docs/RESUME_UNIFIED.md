# Resume: unified batch (launched 2026-09-11 11:37–11:49Z)

**Status at write time:** all four kernels `RUNNING` on 8 T4s. Expected done ~12:50Z.

## 1. Download the four jobs

```bash
cd C:/Users/mnica/Documents/Projet_filiere && for a in maxnicaise:unified-j0-20260911-113738 maxlefrr:unified-j1-20260911-113756 maxnikezz:unified-j2-20260911-114801 maximemonstrenikez:unified-j3-20260911-114953; do acct=${a%%:*}; slug=${a##*:}; KAGGLE_CONFIG_DIR="C:/Users/mnica/.kaggle-accounts/$acct" kaggle kernels output $acct/$slug -p "results/kaggle_outputs/$slug"; done
```

Downloads are slow (~10 min each) and the poller's client-side timeout does **not**
stop the kernel. Check completeness with:

```bash
for d in results/kaggle_outputs/unified-j*/; do echo "$(basename $d): $(find "$d" -name summary.json | wc -l)/12"; done
```

Expect **12 cells per job, 48 total**. If a job reports fewer, read its
`job_summary.json` → `failed_or_incomplete` rather than assuming it finished.

## 2. What this batch is

16 configurations x 3 seeds, **one pinned asset set for every arm**, evaluated
**every epoch**, recording per-epoch mean training loss. This is what makes a
single honest loss-vs-epoch figure possible: the previous overview mixed batches,
and the ablation batch carried a ~0.56 pp offset.

Manifest: `scripts/unified_manifest.py` (groups A references / B Idriss replay /
C new resolution x blur crossings). Driver: `scripts/unified_driver.py`.
Operators: `continuation/ablation_ops.py`, ported unchanged from Idriss's
`continuation-gaussian-tv_exploration_1` @ `1cddcbb`.

Pre-launch checks all passed: `results/unified_verification.json`
(site counts 19 conv / 10 post-ReLU, schedules, finite fwd/bwd, epoch-29 target
path **bitwise** identical to plain ResNet-20).

## 3. Then

* Aggregate as for earlier batches: per-seed values, mean, sample SD, paired
  differences against `plain` **within this batch** (now legitimately paired).
* Rebuild the two English figures with `scripts/plot_all_methods.py` +
  `scripts/method_labels.py`, adding this batch. The batch-offset caveat no
  longer applies *within* it — say so, and keep it for the older batches.
* Key comparisons this batch can finally settle:
  `shrink_b1_relu` vs `shrink_b1` (does blur add anything on top of the best
  reduction?), `shrink_b*_relu_rf` vs `shrink_b*_relu` (does the receptive-field
  profile help?), `shrink_b1_conv` vs `shrink_b1_relu` (placement), and
  `shrink_b2_relu_rf` vs Idriss's 81.14 % (batch offset).

## 4. Standing constraints

Exploratory work on an already-examined test set: no best-epoch selection, no
schedule tuning after seeing results, three seeds give a descriptive SD only.
Do not launch anything further without explicit authorization.

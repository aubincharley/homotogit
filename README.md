# Continuation through image and feature simplification

A small, reproducible research codebase for studying **continuation methods for
neural-network training**, where the continuation parameter controls a
transformation that progressively relaxes a simplification constraint.

The priority is understanding optimization behaviour and transfer to the
original classification problem — not state-of-the-art CIFAR accuracy.

Two distinct families of experiment have been run, and they must not be
conflated:

1. **Input-space continuation** — filter the *images*, anneal toward identity.
2. **Feature-space continuation** — filter the *activations* after every 3×3
   convolution, i.e. the placement used by *Curriculum by Smoothing*.

**[`docs/HANDOVER.md`](docs/HANDOVER.md) is the authoritative record**: every
phase, exact numbers, verification results, hardware gotchas and open threads.
Read it before quoting any past result or re-running anything.

**[`docs/research/continuation/`](docs/research/continuation/README.md) is the
maintained research knowledge base** (in French): the scientific approach, one
record per experiment `EXP-000`…`EXP-011`, operator and protocol definitions,
preserved historical sources with sha256 fingerprints, and an errata list of
interpretations that must not be repeated. Start at its
[README](docs/research/continuation/README.md); an agent picking the project up
should read `CURRENT_STATE.md` and `OPEN_QUESTIONS.md` there first. Note that a
proposal recorded in an archived prompt is **not** authorisation to run it.

## Current state (one-line summary)

| intervention | effect on final test accuracy | cost |
|---|---|---|
| Gaussian **input** blur, annealed | −3.45 pp (GroupNorm, 3 seeds) | negligible |
| Gaussian **feature** smoothing, annealed | **+3.0 pp** (BatchNorm, 3 seeds, full data) | ~1.5× |
| **Progressive input resolution** | **+4.47 pp** (1 seed) | 0.94× |
| Progressive resolution + Gaussian | +4.83 pp vs plain (1 seed) | ~1.4× |
| db2 wavelet feature shrinkage | +0.94 pp (1 seed) — set aside | ~61× |

Current model is `resnet20_bn_cifar` (ResNet-20, BatchNorm, option-A shortcuts,
269,722 params, 19 filter insertion points). Two runs of the same paired plain
configuration differed by 0.04 pp on final accuracy; that is a single observed
difference, **not** a noise floor or an equivalence threshold — two runs define
no distribution (see errata C-12 and C-13 in
[CORRECTIONS](docs/research/continuation/CORRECTIONS.md)). The one-seed rows
above are preliminary.

## Two execution paths

* **`continuation/` + `continuation.cli`** — the original input-space
  experiments (Experiment 0 and Experiment 1), driven by YAML configs.
* **`scripts/continuation_driver.py`** — the current feature-space and
  progressive-resolution campaigns, driven by thin `scripts/job_*.py` configs and
  launched on Kaggle GPUs via [`scripts/kaggle_run.py`](scripts/kaggle_run.py)
  (see [`docs/kaggle_cli.md`](docs/kaggle_cli.md)).

Campaign results land under `results/kaggle_outputs/<kernel-slug>/`; the metrics,
summaries, configs and pairing proofs are committed, the checkpoints are not
(regenerable from the configs and verified seeds).

## Three objects, kept separate

| Object | Where | Changing it means |
|---|---|---|
| Transformation family $T_\eta$ and its native parameter | `continuation/transforms/` | a different *family of objectives* |
| Continuation schedule / controller | `continuation/schedules.py` | a different *traversal of the same family* |
| Optimizer and its LR schedule | `continuation/optim.py` | neither of the above |

The native parameter is family-specific and the framework **never assumes it
decreases toward zero**: Gaussian smoothing has $\sigma\ge0$ with the target at
$\sigma=0$; a relative complexity budget would have $t\in[0,1]$ with the target
at $t=1$. Each family declares its own `target_parameter`.

All transformations are **label-blind**, **deterministic** given their
configuration and the image, and are applied to the **original image**, never to
a previously transformed one — so a later, more detailed stage can always
recover information an earlier stage discarded.

## Layout

```
continuation/
  config.py        strict YAML-backed configuration (unknown keys are errors)
  seeding.py       independent named RNG streams (init / batch / split / probe)
  data.py          CIFAR-10/100, stratified split, deterministic batch stream
  transforms/      transformation-family registry
    base.py        ImageTransform interface, TransformResult, cache keys
    gaussian.py    separable Gaussian + explicit reflection-padding fallback
    tv.py          TV-L2 and TV-Hminus1 budget families (previews only)
    wavelet.py     undecimated wavelet shrinkage (reference + fused paths)
  diagnostics.py   TV / MSE measurements + gradient-transition diagnostics
  schedules.py     constant, piecewise, linear-sigma, geometric, heat-time, budget
  models/
    resnet_gn.py       ResNet-20 + GroupNorm (original)
    resnet18_bn.py     CIFAR ResNet-18 + BatchNorm (reference-style)
    resnet20_bn.py     ResNet-20 + BatchNorm, corrected init  <-- current
  optim.py         SGD + LR schedule indexed by global step
  pipeline.py      uint8 -> float[0,1] -> [resize] -> T_eta -> normalization
  engine.py        Trainer / evaluator at any fixed transformation parameter
  metrics.py       JSONL logging, environment capture, checkpoints
  experiments/exp0.py  fixed-level sweep + aggregation
  experiments/exp1.py  warm-start / continuation branching + aggregation
  viz.py           transformation figures; plotting.py  result figures
  cli.py           command line
scripts/
  continuation_driver.py       current campaign driver (feature space + resolution)
  kaggle_run.py                Kaggle launcher (ships repo, provenance, tags)
  job_*.py                     thin per-experiment configs
  verify_*.py                  operator / placement / pairing checks
  plot_*.py                    campaign figures
configs/           experiment configuration
docs/              HANDOVER.md (read first), gaussian.md, extensions.md,
                   tv_budget.md, wavelet_shrinkage.md, experiment0.md,
                   results.md, exp1_warmstart.md, kaggle_cli.md
  research/continuation/   maintained research knowledge base (French):
                   experiment records, operators, protocols, errata, sources
tests/             ~170 tests, no dataset download required
```

## Install and run

```bash
py -m pip install -r requirements.txt
py -m pytest tests -q
```

```bash
py -m continuation.cli prepare-data --config configs/exp0_gaussian.yaml
py -m continuation.cli visualize    --config configs/exp0_gaussian.yaml
py -m continuation.cli exp0         --config configs/exp0_gaussian.yaml
py -m continuation.cli report       --config configs/exp0_gaussian.yaml
```

Single run at one level:

```bash
py -m continuation.cli train --config configs/exp0_gaussian.yaml --level 1.0 --seed 0
```

Any config value can be overridden, e.g.
`--set optim.total_steps=2000 --set run.device=cpu`.

Switching to CIFAR-100 needs no pipeline change: `--set data.dataset=cifar100`
(the classifier's output width follows the dataset).

## Experiments

### Experiment 0 — fixed Gaussian levels

Independent models trained from scratch at fixed $\sigma\in\{0,0.5,1,2,3\}$
pixels, 3 paired seeds, one shared update budget, everything else held fixed.
Protocol, recorded metrics and interpretation limits:
[`docs/experiment0.md`](docs/experiment0.md). Findings:
[`docs/results.md`](docs/results.md).

At every shared checkpoint each run records, as eval-mode metrics on fixed
subsets: transformed training CE/accuracy, transformed validation CE/accuracy,
**target** (original-image) training CE/accuracy at the same weights, and target
validation CE/accuracy — plus transformation statistics (reconstruction MSE,
retained TV ratio) and separately-accounted transformation cost. Training-mode
minibatch losses are logged under a different record type and are never mixed
with the eval-mode metrics.

### Experiment 1 — warm starts and an intermediate continuation stage

Three arms at the same total budget B = 14,040 and the same paired seeds:
**A** (sigma=0 throughout), **W** (sigma=1 for 1,500 updates, then 0), and
**P** (sigma=1, then 0.5 for 1,500 updates, then 0).  W and P branch from the
same complete sigma=1 state at update 1,500; arm A is reused unchanged from
Experiment 0.  Protocol, checks and findings:
[`docs/exp1_warmstart.md`](docs/exp1_warmstart.md).

```bash
py -m continuation.cli exp1        --config configs/exp1_warmstart.yaml
py -m continuation.cli exp1-report --config configs/exp1_warmstart.yaml
```

Branching preserves parameters, optimizer state (momentum is **carried**), the
global update counter, the sampler position and RNG state, so a branched run
reproduces uninterrupted training exactly and sees the same minibatch indices as
the other arms at equal global updates.  The learning-rate schedule is a pure
function of the global update, so a 1,500-update prefix runs on the full
14,040-update horizon rather than a compressed one.

The official 10,000-image test set is untouched by Experiments 0 and 1. From the
full-data campaign onward it is used deliberately as the reported held-out set
and labelled **test**, not validation; this is exploratory monitoring, with the
protocol frozen before results are examined and no best-epoch selection.

### Feature-space and progressive-resolution campaigns

Run through `scripts/continuation_driver.py`, one `scripts/job_*.py` per
experiment, launched on Kaggle T4s:

```bash
py scripts/kaggle_run.py scripts/job_progressive_resolution.py --gpu \
  --accelerator NvidiaTeslaT4 --dataset pankrzysiu/cifar10-python \
  --include continuation --include scripts
```

Filtering acts at the 19 main-path 3×3 convolution outputs, before
normalization; inputs, shortcuts, pooled vectors and logits are never filtered.
Arms are **paired by verification, not by seed number**: a job embeds sha256
digests of the reference campaign's subset, probe indices, per-epoch
permutations and initial weights, and aborts before training unless they
reproduce exactly (`pairing_verification.json`).

Every evaluation records two explicitly named paths — the **current path** (the
configuration the weights and BN buffers were actually trained under, primary)
and the **target path** (original 32×32, filters bypassed, diagnostic). While
continuation is running the target path measures a premature configuration
change including BatchNorm mismatch, and must not be read as predictor quality.

## What this codebase does not claim

* Not that the discrete, truncated, reflection-padded filter satisfies the
  continuous heat semigroup identity (it measurably does not).
* Not that visual smoothing strictly decreases Shannon information — an ideal
  Gaussian multiplier is nonzero at every finite frequency and, without
  quantization or noise, the map can remain injective.
* Not that lower TV or a lower coding rate makes classification easier.
* Not that Experiment 0 says anything about attraction basins, or that it
  substitutes for actually testing a warm start on the target loss.

A negative result would be scientifically informative: it would not invalidate
every continuation method, just as a positive result would not establish a
general convergence theorem.

## References

* Sinha, Garg & Larochelle, [*Curriculum by Smoothing*](https://papers.nips.cc/paper_files/paper/2020/file/f6a673f09493afcd8b129a0bcf1cd5bc-Paper.pdf),
  NeurIPS 2020 (esp. §3 and Table 7) — internal **feature** smoothing, with
  input-smoothing ablations. It does not justify assuming input blur will help,
  and its intervention is deliberately kept distinct from this input-only baseline.
* Rudin, Osher & Fatemi, [*Nonlinear total variation based noise removal
  algorithms*](https://www.eng.utah.edu/~cs7640/readings/PhysicaRudinOsher.pdf),
  1992 — foundational for TV fidelity/regularity trade-offs in restoration. The
  budget-indexed classification curriculum proposed here is an experimental
  proposal, not a training guarantee from that paper.
* Girod, Stanford EE398A, [*Rate Distortion Theory*](https://web.stanford.edu/class/ee398a/handouts/lectures/04-RateDistortionTheory.pdf)
  — for the distinction between a source rate–distortion function and a
  practical codec's measured rate and distortion.

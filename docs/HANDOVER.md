# Project handover — continuation through image / feature simplification

**Read this first after any context reset.** It records what exists, what was
measured, what was decided, and where to look. Newest work is most detailed.

Repo root: `C:\Users\mnica\Documents\Projet_filiere`
Remote: `https://github.com/aubincharley/homotogit`, branch `continuation-gaussian-tv`
Kaggle account: `maxnicaise` (phone-verified; GPU + internet gated on that).

---

## 0. Research question

Does progressively relaxing a *simplification constraint* during training help
optimization or generalization? Two distinct families of experiment emerged:

1. **Input-space continuation** (phases 1–3): filter the *images*, anneal the
   filter toward identity. **Result: negative.**
2. **Feature-space continuation** (phases 6–9, current): filter the *activations*
   after every 3×3 convolution, anneal toward identity — i.e. the Curriculum by
   Smoothing (CBS) placement. **Result: positive and reproducible.**

These are different interventions and must not be conflated. The paper for (2) is
Sinha, Garg & Larochelle, *Curriculum by Smoothing*, NeurIPS 2020
(`arxiv.org/abs/2003.01367`, code `github.com/pairlab/CBS`).

---

## 1. Codebase map

```
continuation/
  config.py        strict YAML dataclasses; unknown keys raise
  seeding.py       independent named RNG streams (init/batch/split/probe)
  data.py          CIFAR-10/100, stratified split, BatchIndexStream (+state_dict)
  transforms/
    base.py        ImageTransform ABC: target_parameter, cache_key, TransformResult
    gaussian.py    separable Gaussian + explicit reflection padding fallback
    tv.py          TV-L2 and TV-Hminus1 budget families (Chambolle-Pock)
    wavelet.py     undecimated wavelet shrinkage (reference + fused paths)
  diagnostics.py   TV, MSE, retained contrast, gradient-transition diagnostics
  schedules.py     constant / piecewise / linear / geometric / heat-time / budget
  models/
    resnet_gn.py       ResNet-20 + GroupNorm (original)
    resnet18_bn.py     CIFAR ResNet-18 + BatchNorm (reference-style)
    resnet20_bn.py     ResNet-20 + BatchNorm, corrected init  <-- CURRENT
  optim.py         SGD + LR schedule indexed by GLOBAL update
  pipeline.py      uint8 -> float[0,1] -> T_eta -> channel normalization
  engine.py        Trainer for the INPUT-space experiments (exp0/exp1)
  experiments/exp0.py, exp1.py
scripts/
  continuation_driver.py     <-- CURRENT driver (feature-space campaigns)
  kaggle_run.py              Kaggle launcher (ships repo, provenance, tags)
  job_*.py                   thin per-phase configs
  audit_gaussian_placement.py, verify_grad_accumulation.py
  tv_previews.py, wavelet_previews.py, wavelet_benchmark.py, ...
docs/    gaussian.md, extensions.md, tv_budget.md, wavelet_shrinkage.md,
         experiment0.md, results.md, exp1_warmstart.md, kaggle_cli.md, HANDOVER.md
tests/   ~170 tests; `py -m pytest tests -q`
```

---

## 2. Phase history and results

### Phase 1 — Experiment 0: fixed Gaussian *input* blur
`docs/results.md`, `results/exp0_gaussian/`

15 runs: sigma in {0, 0.5, 1, 2, 3} px x 3 paired seeds, ResNet-20/GroupNorm,
45,000 train / 5,000 val, 14,040 updates (40 epochs), no augmentation.

| sigma | target val acc | transformed val acc | retained TV |
|---|---|---|---|
| 0 | 0.8411 | 0.8411 | 1.000 |
| 0.5 | 0.8255 | 0.8355 | 0.820 |
| 1 | 0.7133 | 0.8063 | 0.557 |
| 2 | 0.4832 | 0.7430 | 0.351 |
| 3 | 0.3885 | 0.6741 | 0.255 |

Findings: blurring did **not** speed optimization (sigma=0 descends fastest);
every arm reached ~100% train accuracy, so there was no optimization failure to
fix; coarse models **peak early then decay** on target accuracy (sigma=2 peaks
0.577 at update ~6,333, ends 0.483).

### Phase 2 — Experiment 1: warm starts
`docs/exp1_warmstart.md`, `results/exp1_gaussian_warmstart/`

Three arms at B=14,040, 3 paired seeds: A = sigma 0 throughout; W = sigma 1 for
1,500 updates then 0; P = sigma 1 -> 0.5 -> 0. W/P branch from one complete
sigma=1 state at update 1,500.

Result: **W − A = −1.17 pp, P − A = −0.91 pp, P − W = +0.26 pp (sd 0.45, sign
flips).** Recovery after switching is fast, but neither arm catches up. Infra
built here: `full_state_v1` checkpoints (params, optimizer, global step, sampler
position, RNG), verified exact resume, and a transition-invariance check
(target metrics must not depend on the active sigma) that passed with max |Δ| = 0.

Also added `docs/results.md` "Interpretation note": five categorical claims from
Experiment 0 that the data did not support.

### Phase 3 — TV-budget families (**previews only, never trained**)
`docs/tv_budget.md`, `results/tv_previews/`, `continuation/transforms/tv.py`

Two families sharing one constraint set
`K_t(x) = {z in [0,1]: TV(z) <= t TV(x), per-channel means preserved}`:

* **TV-L2**: minimize `½||z−x||²`
* **TV-Ḣ⁻¹**: minimize `½ (z−x)ᵀ L⁺ (z−x)`, homogeneous form, `L = D1ᵀD1 + D2ᵀD2`,
  `L⁺` the Moore–Penrose pseudoinverse (zero on the constant mode). Applied
  exactly via DCT-II diagonalization of the Neumann Laplacian.

Solver: Chambolle–Pock with `K = [D; I]`, so the **relative budget is a hard
constraint** — no TV penalty coefficient anywhere (a shared lambda would not be
equivalent across images). Native parameter `t in [0,1]`, target endpoint at
**t = 1** (not 0) — this is why `ImageTransform` exposes `target_parameter`.

Preview diagnostics (10 images, t in {1, .9, .75, .5, .25, 0}):

| t | TV ratio L2 / Ḣ⁻¹ | retained contrast rho L2 / Ḣ⁻¹ |
|---|---|---|
| 0.9 | 0.900 / 0.900 | 0.984 / 0.989 |
| 0.5 | 0.500 / 0.500 | 0.869 / 0.906 |
| 0.25 | 0.250 / 0.251 | 0.698 / 0.759 |

Ḣ⁻¹ retains more contrast at equal TV. Cost: 1,113 s (L2) and 2,222 s (Ḣ⁻¹) for
10 images x 6 budgets on CPU, ~10^4 PDHG iterations each. One solve
(Ḣ⁻¹, t=0.25, the airplane) stayed 3.8% over budget after 40,000 iterations —
reported, not hidden. A bug found here: a **sign error in the adjoint Dᵀ**,
caught by `<Dz, g> == <z, Dᵀg>`.

**Never used for training. On hold.**

### Phase 4 — Wavelet shrinkage
`docs/wavelet_shrinkage.md`, `results/wavelet_previews/`,
`continuation/transforms/wavelet.py`

See §5 below for the formula. Verified: `W*W = I` to 1e-12 (tested directly, not
via any identity shortcut), adjoint and Parseval identities, band energies match
`pywt.swt2(norm=True)` to 1e-13, gradcheck in float64, constants preserved.

Cost vs Gaussian: **14–36x on CPU per image; ~30–80x on GPU**. Optimized later
with a fused grouped-convolution path (2.0–2.1x forward for db2/sym4/coif1,
1.4x fwd+bwd; haar slightly slower) plus an identity bypass at s=1
(209 ms -> 0.17 ms, 1911 -> 20 MiB). Gradient checkpointing measured **no memory
win** and was dropped. In-network at microbatch 4 with 19 insertions: 163.7 ms
and 578 MiB vs plain 10.1 ms / 24 MiB.

**db2 continuation prepared but never launched** — `scripts/job_db2_study.py`,
`s_k = 1 − 0.5·max(1 − k/600, 0)`. **On hold.**

### Phase 5 — Kaggle infrastructure
`docs/kaggle_cli.md`, `scripts/kaggle_run.py`

Hard-won facts:
* `machine_shape` **is** the accelerator selector (`request.machine_shape = acc or
  metadata["machine_shape"]`). Pass `--accelerator NvidiaTeslaT4`. **Kaggle does
  not validate it** — a garbage name is accepted silently, so acceptance proves
  nothing.
* GPU **and** internet are gated on **phone verification**. Before verifying,
  every worker came up CPU-only (`torch 2.10.0+cpu`, no `nvidia-smi`) and DNS
  failed. After verifying: 2x Tesla T4, 15,360 MiB, `torch 2.10.0+cu128`.
* Kernels have **no internet**; CIFAR-10 comes from the attached dataset
  `pankrzysiu/cifar10-python`, mounted at
  `/kaggle/input/datasets/<owner>/<slug>/cifar-10-batches-py` (discovered by
  recursive glob, not assumed). `download=False` makes torchvision verify the
  official md5s.
* The runner ships the `continuation` package + `scripts/` into the kernel as a
  base64 zip, writes `environment.json` provenance, tags slugs per run so nothing
  is overwritten, pulls logs on failure, and defaults internet **off**.
* CLI subcommand is `logs`, not `log`; `kernels output` already retrieves the log.
* `kaggle` may not be on PATH; the runner resolves it via `shutil.which` ->
  Scripts dir -> `python -m kaggle`.

Usage: `py scripts/kaggle_run.py <job.py> --gpu --accelerator NvidiaTeslaT4
--dataset pankrzysiu/cifar10-python --include continuation --include scripts
--title <name>`

### Phase 6 — the pivot: feature-map continuation

Input-space continuation was negative, so the intervention moved to **activations
at the 19 (ResNet-20) / 17 (ResNet-18) main-path 3x3 convolution outputs**, i.e.
the CBS placement. Attached by forward hooks on `Conv2d` with
`kernel_size == (3,3)`; 1x1 projection shortcuts are excluded automatically.

Ordering (verified in source and by execution trace):
```
stem / first block conv:  conv -> Gaussian -> Norm -> ReLU
second block conv:        conv -> Gaussian -> Norm -> + shortcut -> ReLU
```
Inputs, shortcuts, pooled vectors and logits are never filtered.

### Phase 7 — LR diagnosis and the 10,000-image studies
`results/kaggle_outputs/lr-diagnostic-*`, `plain-study-*`, `gaussian-study-*`,
`lr-control-002-*`

The first pilot **collapsed to uniform prediction** (CE = ln 10 = 2.3026) because
LR 0.1 (tuned for 45k images / 14,040 updates / 400 warmup) was far too high for
600 updates with 30 warmup. Diagnostic: LR 0.02 -> final probe CE 1.888,
**LR 0.005 -> 1.587 (selected)**. Both far below ln 10.

Gradient accumulation was verified first and is **exact**: 3.97e-16 relative error
in float64. The ~6e-4 in float32 on CUDA is different execution paths across batch
sizes (norms agree to 7e-6; deterministic flags don't change it; same-path-twice
is 5e-8). **No bug; nothing invalidated.** `results/grad_accumulation_check.json`

10,000 images, 1,200 updates, ResNet-20 **GroupNorm**, 3 paired seeds:

| arm | test/val acc |
|---|---|
| plain | 0.5036 ± 0.0052 |
| Gaussian | 0.4691 ± 0.0060 |

**Gaussian − plain = −3.45 pp (sd 0.57)**, negative in all seeds; also −2.90/−2.78
pp at LR 0.005/0.02 in the seed-0 2x2 control. So with **GroupNorm**, feature
smoothing *hurt*.

### Phase 8 — the ResNet-18 / ResNet-20 architecture hesitation

To test whether the negative result was architecture-specific, a reference-style
**CIFAR ResNet-18 + BatchNorm** was built (`resnet18_bn_cifar`, 11,173,962 params,
17 insertions, `fan_in` conv init, `N(0, 0.01²)` classifier). Same protocol,
seed 0:

| arm | test acc |
|---|---|
| plain | 0.5426 |
| Gaussian | **0.6216** |

**+7.90 pp — the sign flipped.** Architecture and initialization changed together,
so this did not isolate the cause. A genuine blocker appeared here: ResNet-18's
stage-4 maps are **4x4** and our kernel has radius 4, but PyTorch's native reflect
padding requires `pad < n`. Resolved (see §4) by explicit whole-sample reflection
indexing, keeping the 9-tap kernel and all insertions.

An **audit** (`scripts/audit_gaussian_placement.py`,
`results/gaussian_placement_audit.json`) traced a real forward/backward and
confirmed: 269,722 params, 19 insertions at (16,32,32)/(32,16,16)/(64,8,8), filter
normalized/channelwise/shape-preserving, gradients flow through it, schedule
identical across insertion points, and the k>=600 bypass reproduces plain
**bitwise** (max grad diff 0.0).

A difference table vs `pairlab/CBS @ 5f62e7da` is in the conversation record;
essentials: reference uses ResNet-18 + BatchNorm + learned 1x1 projection
shortcuts, a **non-separable 3x3** Gaussian with **zero** padding rebuilt once per
epoch, `sigma_0 = 1` x0.9 every 5 epochs (**never reaching 0**), batch 64,
200 epochs, LR 0.1 with /10 at epochs 30/60/90, Normalize(0.5,0.5,0.5), and
`fan_in` + `N(0,0.01²)` init. All reference values are **repository defaults**,
not settings documented for the published numbers.

### Phase 9 — ResNet-20 + BatchNorm pilot
`results/kaggle_outputs/resnet20bn-gaussian-20260908-154226/`,
`results/resnet20bn_pilot_corrected.png`

`resnet20_bn_cifar`: ResNet-20 structure (widths 16/32/64, option-A shortcuts,
269,722 params) with BatchNorm (momentum 0.1) and reference-style init. Seed 0,
10,000 images, 2,400 updates, piecewise sigma 1.00 -> 0.30 over 1,700 updates
then 700 unfiltered.

**Gaussian − plain = +4.26 pp (0.5962 vs 0.5536), val CE −0.126.**

**Critical presentation lesson:** while filters are active, the *bypassed*
evaluation is meaningless for BatchNorm models — it showed −32 pp at update 600
purely from running-statistic mismatch. With **active-filter** curves as primary,
Gaussian leads at *every* checkpoint (+5.80 pp at update 200). Always plot
active-filter as primary and bypassed as a thin dashed diagnostic.

### Phase 10 — CURRENT: full-data campaign, 9 runs
`results/kaggle_outputs/fulldata-r20bn-20260908-161221/`,
`results/fulldata_campaign.png`, `scripts/job_fulldata_campaign.py`

Official CIFAR-10: **all 50,000 train / 10,000 test** (the old 5,000 val split is
now part of training; curves are labelled **test**). 30 epochs = 11,730 updates
(391/epoch incl. a partial final batch of 80). ResNet-20 + BatchNorm, corrected
init, 19 insertions. Effective batch 128 as microbatches of 32, each weighted by
its actual example count. SGD LR 0.005, momentum 0.9, wd 5e-4, 60 warmup then
cosine. No augmentation. Normalization stats from the full 50,000. Seeds 0/1/2
share init weights, BN buffers and per-epoch permutations across arms.

Arms (e = zero-based epoch):
* **plain** — no filtering
* **plateau** — 1.00/0.85/0.70/0.60/0.50/0.40/0.30 in 3-epoch blocks, bypass from e=21
* **compressed geometric** — `sigma = 0.9**e` for e<=20, bypass from e=21
  (paper's sigma_0 and factor, one-epoch interval, explicit final bypass —
  **not** an exact CBS reproduction)

**Results at epoch 30 (frozen; no best-epoch selection):**

| arm | test acc mean ± sd | test CE | per-seed acc |
|---|---|---|---|
| plain | 0.7540 ± 0.0040 | 0.7615 ± 0.0039 | 0.7510 / 0.7526 / 0.7585 |
| plateau | **0.7841 ± 0.0032** | **0.6259 ± 0.0094** | 0.7827 / 0.7878 / 0.7819 |
| geometric | **0.7846 ± 0.0072** | 0.6636 ± 0.0255 | 0.7763 / 0.7884 / 0.7892 |

Paired vs plain: plateau **+3.01 pp** (sd 0.61), geometric **+3.06 pp** (sd 0.53),
both positive in all three seeds. plateau − geometric = +0.05 pp (sd 0.69):
indistinguishable on accuracy, plateau better and less variable on CE.

Trajectory: filtered arms trail plain for ~14 epochs, cross over at epochs 16–18,
finish ahead; no discontinuity at the epoch-21 bypass. Runtimes 458 s (plain) /
~690 s (filtered) per run; 5,504 s total; peak 511–584 MiB.

---

## 3. Headline conclusion so far

**Input-space** Gaussian continuation: negative (−3.45 pp with GroupNorm,
−1.17 pp for warm starts). **Feature-space** Gaussian continuation with
**BatchNorm**: consistently positive (+3.0 pp on full data, 3 seeds; +4.3 pp and
+7.9 pp in pilots). The GroupNorm-vs-BatchNorm contrast is the most interesting
unresolved variable — but data scale, duration and schedule changed alongside it,
so **the cause is not isolated**.

---

## 4. Numerics worth remembering

* **Explicit reflection padding** (`continuation/transforms/gaussian.py`):
  `P = 2(n−1)`, `m = i mod P`, `r_n(i) = min(m, P−m)` for `i = −pad .. n+pad−1`.
  For `n=4, pad=4`: `[2,3,2,1,0,1,2,3,2,1,0,1]`. Native `F.pad(mode="reflect")`
  is used when `pad < n`, this gather otherwise; both differentiable, identical
  where both apply. Needed for radius-4 kernels on 4x4 feature maps.
* Gaussian: separable, 9 taps, radius 4, normalized, channelwise (`groups=C`),
  exact identity object at sigma=0.
* WDDM/Windows silently oversubscribes VRAM into host memory instead of raising
  OOM — a probe once reported 8,712 MiB "allocated" on a 4,096 MiB card and 67 s
  per step. Always gate fit-tests on measured peak vs physical VRAM.
* Evaluate in `eval()` and restore the prior mode; never update or recalibrate BN
  statistics during probes.

---

## 5. Wavelet shrinkage — formula and principle

Two-level, **separable, undecimated (stationary/à trous)** wavelet transform,
applied independently per sample and channel, at each insertion point. Boundary:
mirror the input to `2H x 2W`, run the *periodic* transform there, crop the
original top-left block back out. Filters scaled by `1/sqrt(2)` so the analysis is
a **tight frame with bound one** (`W*W = I`); synthesis is the exact adjoint `W*`.

Analysis keeps the coarsest approximation and all six detail bands:
```
W E h = ( a_2 , { d_{j,o} } ),   j in {1,2},  o in {LH, HL, HH}
```

Per sample, channel and band, with `N` the number of spatial coefficients:
```
nu_{j,o}     = ||d_{j,o}||_2 / sqrt(N)                      (plain RMS)
lambda_{j,o} = 4 (1 - s) 2^(1-j) nu_{j,o}
T_s(h)       = P W* ( a_2 , { soft(d_{j,o}, lambda_{j,o}) } )
soft(v, l)   = sign(v) max(|v| - l, 0)
```

Principle: **the approximation band is never touched**; only the detail bands are
soft-thresholded, with a threshold proportional to that band's own RMS, so it
adapts per image/channel/band rather than using a global constant. The factor
`2^(1-j)` thresholds the finer level (j=1) twice as hard as the coarser (j=2).
`s` is the continuation parameter: `s = 1` gives zero threshold and hence exactly
the identity (`T_1 = P W* W E = I`); smaller `s` removes more detail. `T_0` is
**not** constant — at s=0 it still retains ~6% of the distance to the coarse-only
reconstruction. Thresholds are differentiable through `nu`, with
`nu = sqrt(mean(d²) + eps)` giving a finite (zero) gradient on all-zero bands.

Wavelets compared: `haar`, `db2`, `sym4`, `coif1`. It is an explicitly defined
shrinkage operator — **not** a constrained-TV solution and **not** a claimed
proximal operator for redundant wavelet analysis.

---

## 6. On hold / open threads

* **db2 wavelet continuation** — configured, never launched
  (`scripts/job_db2_study.py`). Estimated 45–70 min for 3 seeds.
* **TV-L2 / TV-Ḣ⁻¹** — previews only; far too slow (~10^4 iterations/image) for
  on-the-fly training. Would need offline caching per budget.
* **CBS reference reproduction** — configuration specified (ResNet-18, BN,
  physical batch 64 since accumulation does not reproduce BN statistics, 200
  epochs, LR 0.1 with /10 steps); estimated 7–13 h/arm, never run.
* **Isolating GroupNorm vs BatchNorm** at fixed data scale, duration and schedule.
* Official test set was untouched until phase 10, where it became the reported
  held-out set by design.

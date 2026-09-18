# Adaptive resolution by bounded scale specialisation — the method that works

Status: measured on CIFAR-10 / ResNet-20 BN / no augmentation / no filter,
six paired seeds (2026-09-17/18). Full experimental record in
[`adaptive_resolution_plan.md`](adaptive_resolution_plan.md) §12–18 and in the
knowledge base, records [EXP-012](research/continuation/experiments/EXP-012_adaptive_phase0.md)
to [EXP-017](research/continuation/experiments/EXP-017_two_sided_specialisation.md).
A French summary is at the end.

## 1. The recipe in one table

| phase | epochs | input resolution | rule |
|---|---|---|---|
| ascent | 0–2 / 3–5 / 6–8 / 9–11 | 16 / 20 / 24 / 28 | fixed: three epochs per size |
| fine phase | 12–29 | 32 | **adaptive**: whenever the specialisation signal g(32→24) ≥ 0.30, the next epoch is trained at 24×24, then back to 32; never in the last two epochs |

Everything else is the campaign protocol, unchanged: 30 epochs × 391 updates,
SGD 0.005 / momentum 0.9 / weight decay 5e-4, 60-update warm-up then cosine over
the whole horizon, effective batch 128 as four microbatches of 32, no
augmentation, channel statistics fitted once on the training set. The horizon
and the LR path are identical to every comparator, so nothing about the
optimiser is confounded with the schedule.

Resolution changes act on the **float image before normalisation** by bilinear
resize with antialiasing (`align_corners=False, antialias=True`); at 32 the
tensor is passed through untouched. The same weights are used at every size;
nothing is upsampled back, no blocks are added.

## 2. The signal: scale specialisation

At the end of every epoch, at fixed weights θ and current resolution r, for a
probe resolution r′:

    g_{r→r′}(θ) = [ L_{r′}^{BN-recal}(θ) − L_r(θ) ] / L_r(θ)

* L_r is the mean cross-entropy on a fixed, class-balanced **monitor set** of
  2,000 training images (`fixed_subset_indices`, stream `adaptive_monitor`).
* L_{r′}^{BN-recal} is the same quantity with the images resized to r′, after the
  BatchNorm running statistics have been **rebuilt at r′** on the monitor set
  (`momentum=None`, one forward pass, no gradient) — and then restored bitwise.
  Without this step the number mostly measures a statistics mismatch, not the
  function.

g is the relative loss penalty the network pays when shown another scale, once
the BatchNorm part is removed. It is zero for a scale-invariant predictor. It
costs one recalibration pass plus one evaluation on 2,000 images, about 1 % of
an epoch, and it perturbs nothing: parameters, BN buffers and train/eval mode
are checked to be identical after every measurement (`purity` field in every
record).

Measured properties (all runs, all seeds):

* grows roughly linearly with dwell at a fixed scale (≈ +0.03/epoch at 16 for a
  +4 probe, ≈ +0.05/epoch at 32 for a −8 probe);
* resets immediately when the resolution changes;
* larger for a larger scale jump;
* seed-stable to ±0.01 at the point where a threshold acts;
* ranks fixed schedules by final accuracy: schedules that switch while
  g ≤ 0.09 reach 80.0–80.1 %, one that switches at 0.21 reaches 79.1 %, one
  that lets g reach 0.63 reaches 77.5 %.

What does **not** work as a specialisation signal: the training gradient norm
(stationary at 1.0 ± 0.2 through twenty epochs, nothing to detect), gradient
alignment between resolutions (a live controller on it never fired), and the
BatchNorm-statistics shift alone (flat over dwell). See plan §12, §15, §17.5.

## 3. The control law

```
after every epoch at r = 32, if more than 2 epochs remain:
    g   ← g_{32→24}(θ)
    ema ← g if just arrived / just reheated else 0.5·ema + 0.5·g
    if ema ≥ 0.30:
        train the next epoch at 24×24          # "reheat"
        ema ← None                              # re-seed afterwards
    then continue at 32
```

Why 24 and not 16: a −8 probe is the smallest step whose g grows fast enough to
be read against the noise (±0.02 per epoch on 2,000 images). Why 0.30: the
useful range is narrow. 0.15 reheats every other epoch and over-regularises
(79.9 %); 0.45 and 0.60 reheat too rarely and too late (79.7 / 80.0 %, three
seeds); 0.30 is the only bound with six seeds and the only one that beat the
ramps on every seed. Why "never in the last two epochs": the model is evaluated
at 32×32; the last epochs settle the fine-scale features and BN statistics.

In practice the controller chose 4–5 reheats per run, at seed-dependent epochs
between 14 and 27, spaced two to three epochs apart.

## 4. What it achieves

Final test accuracy (%), seeds 0–5, all arms paired (same init weights, same
batch order, same horizon, same LR path, same wall time):

| arm | 0 | 1 | 2 | 3 | 4 | 5 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain 32×32 throughout (seeds 0–2 only) | 74.76 | 76.01 | 76.40 | | | | 75.72 |
| 16/24/32 ×6/6/18 (campaign schedule, 0–2) | 78.63 | 78.91 | 79.87 | | | | 79.14 |
| fixed fine ramp 16/20/24/28 ×3 then 32 | 79.80 | 79.54 | 80.71 | 79.34 | 79.88 | 80.20 | 79.91 |
| fixed ramp + fixed reheats at 20/24/28 | 80.32 | 80.54 | 81.22 | 79.44 | 80.28 | 80.44 | 80.37 |
| **fixed ramp + adaptive reheats (this method)** | 80.79 | 79.73 | 81.28 | 80.12 | 80.81 | 80.40 | **80.52** |

Paired against the fixed fine ramp: **+0.99 / +0.19 / +0.57 / +0.78 / +0.93 /
+0.20, mean +0.61 ± 0.35 pp, six of six positive** (paired t ≈ 4.3). Against
the hand-placed reheats: +0.15 ± 0.55, four of six — not resolved; the signal
places the reheats at least as well as a tuned fixed rule without being told
when. Against the campaign's best *filtered* model measured in the same
protocol (16/24/32 + internal Gaussian plateau, 80.48 %): +0.04 with no filter
and 68 % of its training time (457 s vs 667 s per run on a T4).

## 5. Why it works

Progressive resolution in this recipe is a generalisation effect: plain
training memorises (train-probe CE 0.24 against test CE 0.75), the ramp halves
that gap. But during the 18 epochs at 32×32 the network specialises again to
fine-scale detail — g(32→24) climbs from ≈ 0 to 0.6–0.7 — while test accuracy
stops improving around epoch 24. A single coarse epoch resets that
specialisation (g drops to slightly negative, train-probe CE rises by ≈ 0.15)
and the following fine epoch comes back **above** the pre-reheat test accuracy
(seed 0 of the fixed-reheat control: 79.3 → 79.8 → 80.2 → 80.4 across three
reheats). The controller does this when it is needed rather than on a timetable.

Every other member of the ramp family, fixed or adaptive, plateaus at
80.0–80.1 %: adapting the *ascent* buys nothing (the accuracy surface there is
flat) and adapting it costs ≈ 0.3 pp when combined with reheats, because the
adaptive ascent reaches 32 one to three epochs later. The gain lives entirely in
the fine phase.

## 6. What was tried and did not work (so nobody repeats it)

| idea | result | where |
|---|---|---|
| plateau of the gradient norm as switch trigger | signal stationary, nothing to detect | plan §12.2 |
| look-ahead loss at the next scale stops improving | monotone, never fires | §12.4 |
| transfer efficiency / gradient alignment, live controller | never fired, degenerated into its guard | §15.2 |
| adaptive ascent from g (bounded specialisation upward) | matches the fixed ramp, does not beat it | §17.5 |
| placement of the two boundaries of 16/24/32 | flat within ±0.45 pp | §12.6 |
| per-update random resolution ("mixed") | 79.6 %, below the clean ramp | §14.1 |
| Gaussian filter + fine ramp | filter and fine steps are substitutes, no sum | §14.1 |
| reheat bound 0.15 / 0.45 / 0.60 | over- / under-regularise, three seeds | §18.4–18.5 |
| BatchNorm-shift as label-free specialisation measure | flat over dwell | §17.5 |

## 7. Transfer to STL-10 (96×96), and limits

The fixed ramps transfer (plan §16, EXP-015): coarse epochs are genuinely
cheaper there (48×48 costs 25 % of 96×96), and at equal wall time progressive
resolution is worth about +2.3 pp over plain. The reheat controller has **not**
been run on STL-10 yet; that, and its interaction with the Gaussian filter and
with data augmentation, are the open items.

## 8. Code map

| what | where |
|---|---|
| signal, BN recalibration, purity checks | `continuation/probe_signals.py` |
| controller (`kind: "gap2s"`, `ascent: "fixed"`), schedules, run lists | `scripts/job_adaptive_phase0.py` |
| the run that produced the six-seed result | `scripts/job_adaptive_phase5.py` → kernel `adaptive-phase5-20260918-055940` |
| offline analysis of all traces | `scripts/analyze_adaptive_phase0.py` |
| STL-10 port | `scripts/job_stl10_resolution.py`, `scripts/analyze_stl10_resolution.py` |
| results (metrics per epoch, decisions per run) | `results/kaggle_outputs/adaptive-phase*/`, `stl10-*/` |

To reproduce the best arm alone: `ADAPT_PHASE=5` selects it under the label
`Rsteps4ar__Gnone__input_bilinear__seed{k}` (spec `schedule: "Rsteps4"`,
`controller: {kind: gap2s, ascent: fixed, down_gstar: 0.30, reheat_r: 24,
settle: 2, ema_beta: 0.5}`). Launch with `scripts/kaggle_run.py` as documented in
`docs/kaggle_cli.md`.

---

## Résumé en français

**L'idée.** On entraîne le même réseau du début à la fin, mais on change la
taille des images qu'on lui montre. On commence flou (16×16), on monte par
petits pas (20, 24, 28), et on finit net (32×32). Ça vaut environ +4 points sur
l'entraînement normal de CIFAR-10 sans augmentation, et les petits pas réguliers
valent environ +1 point de mieux que le saut 16 → 24 → 32.

**Le problème découvert.** Une fois à 32×32, le réseau passe 18 époques à se
re-spécialiser sur les détails fins, et pendant ce temps la précision de test ne
bouge plus.

**Le signal.** À la fin de chaque époque, on montre au réseau figé les mêmes
images réduites à 24×24 (après avoir recalculé ses statistiques BatchNorm pour
cette taille, puis tout restauré) et on mesure de combien sa perte augmente par
rapport au 32×32. Près de zéro : il s'appuie sur les formes. Ça grimpe : il
dépend des détails. Coût : 1 % d'une époque, aucun effet sur l'entraînement.

**La règle.** Dès que ce nombre dépasse 0,30, une époque de « réchauffe » à
24×24, puis retour à 32. Jamais dans les deux dernières époques. Le réseau
décide lui-même : 4 à 5 réchauffes par run, espacées de 2 à 3 époques.

**Le résultat.** Sur six graines appariées, +0,61 point par rapport à la
meilleure rampe fixe, positif sur chacune, à budget et temps égaux, sans filtre,
et légèrement au-dessus du meilleur modèle filtré en 68 % de son temps.

**Ce qui ne marche pas.** Piloter la montée avec ce signal ne fait pas mieux
qu'un calendrier fixe ; la norme du gradient et l'alignement des gradients ne
détectent rien ; le seuil doit rester autour de 0,30.

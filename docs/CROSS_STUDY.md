# Two studies of the same four methods

Two independent lines of work probe the four frozen `continuation-core`
configurations. They share the methods, the architecture, the dataset and the
pinned assets; they share almost nothing else.

| | loss-landscape study (`landscape_v1-v3`) | endpoint probes (`analysis/`, [`RESULTS_GRID.md`](RESULTS_GRID.md)) |
|---|---|---|
| space | weights | function inputs, plus weight curvature |
| replication | 5 paired seeds | 3 seeds, 20 conditions |
| BatchNorm policies | both, throughout | both (added after this comparison) |
| distinctive tools | 2-D surfaces, interpolation barriers, trajectory PCA, fixed-weight states | input-Jacobian spectrum, radial frequency profile, amplitude curve, Hessian trace and per-block traces, BatchNorm gauge |
| statistics | preregistered analysis choices | four guards, incl. the between/within-group decomposition |

This document records where they agree, where they appeared to conflict, and
what the conflict turned out to be.

---

## 1. The base reproduces across three sources

Test accuracy, mean ± SD over seeds:

| method | endpoint probes (n=3) | landscape v2 (n=5) | [`METHODS.md`](METHODS.md) (n=3) |
|---|---|---|---|
| `plain` | 75.93 ± 1.04 | 75.65 ± 0.78 | 75.43 ± 0.76 |
| `resolution_max_b1` | 80.22 ± 0.01 | 80.21 ± 0.36 | 80.37 ± 0.61 |
| `gaussian_postrelu` | 79.95 ± 0.04 | 80.12 ± 0.49 | 79.91 ± 0.27 |
| `resolution_max_b1_gaussian_conv` | 81.35 ± 0.09 | 81.58 ± 0.30 | 81.51 ± 0.22 |

Means agree to **0.25 pp** across three independent executions, two of them by
different people.

Both studies also find the curricula **fit the training set less**: training
cross-entropy 0.27–0.31 against 0.20 for the control (v2), training error
0.084–0.101 against 0.057 (endpoint probes). The gain is not better
optimisation.

**A correction this comparison forced.** The endpoint probes' seed spreads
(0.006–0.09 for the curricula) are an order of magnitude tighter than both other
sources (0.22–0.61). Two independent datasets contradict them, so no claim about
the curricula reducing seed variance can stand; the cause of the tight spreads in
that run is unexplained.

---

## 2. The dissociation, found twice, in unrelated spaces

Neither study carried this alone.

| | Gaussian | resolution reduction |
|---|---|---|
| **frequency profile of the input-Jacobian** (endpoint probes) | shifted, −8.8 % mean radius | unchanged, +4.6 % |
| **weight-perturbation sensitivity, recalibrated** (landscape v2) | **+43 % to +108 %**, 5/5 seeds | −32 %, 5/5 seeds |

Same accuracy gain, ~+4.5 points each. Two measurement spaces with no shared
machinery, and the Gaussian arm is the singular one in both.

Both studies also reach the same negative conclusion independently: v2 records
that "the accuracy gain does not track recalibrated perturbation sensitivity"
and that "neither is shown to cause the other"; the endpoint probes find that no
measured quantity predicts test error at equal training error across 20
conditions, and that every strong pooled coefficient is a grouping artefact.

---

## 3. The apparent conflict, and what it was

The endpoint probes measured `tr H` **30 % lower** for the Gaussian method — a
flatter neighbourhood. Landscape v2 measured it **43 % more sensitive** under
recalibration. Both are in weight space, so the signs should agree.

Two choices in the endpoint probe were load-bearing and untested: the stored
BatchNorm statistics, and the training split. The 2×2 was therefore measured —
4 methods × 3 seeds × 2 policies × 2 splits, 48 points.

### tr H against the control

| | frozen / train | frozen / test | recalibrated / train | recalibrated / test |
|---|---|---|---|---|
| Gaussian | −30.3 % | −42.6 % | −19.2 % | −21.1 % |
| resolution | −32.0 % | −46.0 % | −28.1 % | −30.5 % |
| combined | −41.0 % | −54.9 % | −36.9 % | −41.6 % |

**The contrast holds in 36 measurements out of 36.** Nothing reverses.

Three findings:

1. **Recalibration does not flip the curvature.** All three methods stay flatter
   than the control under both policies. The `tr H` result was not an artefact of
   stale statistics.
2. **The effect is not a property of the training split.** On test images it is
   *larger*, not smaller: −42.6 % against −30.3 % for the Gaussian method under
   frozen statistics. The warning that the resolution method's advantage
   evaporates on test data applies to finite-amplitude sensitivity, not to
   curvature.
3. **But the policy changes the amount by a factor of 4.1 to 7.9.** That
   reproduces v2's reported 3–7× on `S(ε)` and carries the same reading: much of
   what the frozen policy measures at finite amplitude is a statistics mismatch
   rather than geometry. The curriculum effect survives the deflation but
   shrinks — −30.3 % becomes −19.2 % for the Gaussian method.

### So the two results are not in conflict — they are a third dissociation

| | `tr H` (endpoint probes) | `S(ε)` recalibrated (landscape v2) |
|---|---|---|
| scale | infinitesimal | ε = 0.025 to 0.25 |
| directions | Rademacher, raw coordinates | filter-normalised |
| Gaussian vs control | **−19 % to −21 %** | **+43 % to +108 %** |

The Gaussian solution's curvature *at the point* is lower; its response *at a
finite distance* is higher. Both are true of the same network.

That is the same reversal the endpoint probes already found on the input side —
the Gaussian method leads on the derivative (‖J‖_F −47.8 %) and trails at finite
amplitude (S(3) −12.9 %, against −16.5 % for the resolution method). **Two
independent spaces, the same reversal**, and it matches the measured fact about the
linearity ratio: it is 0.85-0.91 at `eps = 0.05` -- a perturbation smaller than
one 8-bit quantisation step of the image -- and 0.05-0.09 by `eps = 3`.  A
first-order description holds, but only below the scale at which any
neighbourhood argument operates.

---

## 4. What each study corrects in the other

**The landscape study corrects the endpoint probes** on seed dispersion (§1),
and showed that the BatchNorm policy and the evaluation split had to be tested
rather than assumed — which is what §3 does.

**The endpoint probes correct the landscape study** on two points. Its `S(ε)`
is filter-normalised and therefore protected from the BatchNorm rescaling
symmetry, but its surfaces and interpolation barriers are not — the gauge
quotient in [`PROBES.md`](PROBES.md) applies to both. And its reading that
"lower sensitivity and higher accuracy co-occur for two of three methods" is
exactly the situation guard 4 exists for: four methods are four points, and a
pooled coefficient over them cannot be read as a relation.

---

## 5. What the two establish together

1. The gain is reproducible to 0.25 pp across three independent executions, and
   the curricula reach it while fitting the training set *less*.
2. **The two interventions do not do the same thing.** Three dissociations now:
   frequency (blur only), recalibrated weight sensitivity (resolution only), and
   infinitesimal-versus-finite curvature (both spaces).
3. Curriculum solutions are flatter in weight space by 19–55 %, in 36
   measurements out of 36, across both BatchNorm policies and both splits.
4. No quantity in either study explains the gain, by two different protocols.
5. Every PAC-Bayes route tried is closed, including the displaced prior.

And what neither establishes: any causal claim, anything about seed variance,
and anything beyond CIFAR-10, ResNet-20 and this recipe.

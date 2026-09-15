# What a curriculum leaves on the solution

28 cells, 20 conditions, four frozen methods.  No result here required a new
method: the grid keeps `plain`, `resolution_max_b1`, `gaussian_postrelu` and
`resolution_max_b1_gaussian_conv` untouched and varies the *training recipe*
around them — three seeds each, plus four optimiser variants per method.

Twenty conditions were chosen over twelve for one reason.  Four conditions cannot
support a correlation, and reading one anyway is the failure this design avoids.
The variants also span 12.5 points of test accuracy instead of 5.4, which is what
gives the analysis power.

Probes: [`PROBES.md`](PROBES.md).  Figures: `results/grid_figures.png`.
Report: `results/grid_report.json`.  Comparison with the companion
loss-landscape study: [`CROSS_STUDY.md`](CROSS_STUDY.md).

---

## 0. The four methods reproduce

| method | this run, seeds 0/1/2 | [`METHODS.md`](METHODS.md) record | difference |
|---|---|---|---|
| `plain` | 74.73 / 76.51 / 76.56 | 74.58 / 75.66 / 76.05 | +0.50 |
| `resolution_max_b1` | 80.21 / 80.22 / 80.22 | 79.97 / 80.06 / 81.07 | −0.15 |
| `gaussian_postrelu` | 79.91 / 79.95 / 79.98 | 79.60 / 80.06 / 80.06 | +0.04 |
| `resolution_max_b1_gaussian_conv` | 81.30 / 81.30 / 81.45 | 81.30 / 81.50 / 81.73 | −0.16 |

Means reproduce to within 0.5 pp.

**The seed spreads do not.**  Recorded: 0.76 / 0.61 / 0.27 / 0.22.  Here: 1.04 /
0.006 / 0.035 / 0.087.  The curriculum methods are seven to a hundred times
tighter in this run than in the record, and the control is wider.  Seeds are
correctly wired (verified in each `config.json`), and the initialisations are
genuinely distinct (pairwise distance 52.3–52.5).  The discrepancy is
**unexplained**, and no claim about the curriculum reducing seed variance should
be made until it is: three seeds cannot distinguish a real effect from a lucky
draw against a record that says otherwise.

---

## 1. The solutions are flatter, and the gauge does not explain it

| | tr H | ‖J‖_F | S(3) | mean radius | test error |
|---|---|---|---|---|---|
| `plain` | 14730.5 | 143.2 | 12.79 | 12.075 | 0.2407 |
| `gaussian_postrelu` | 10360.8 | 74.8 | 11.14 | 11.007 | 0.2005 |
| `resolution_max_b1` | 10108.2 | 100.0 | 10.67 | 12.633 | 0.1978 |
| `resolution_max_b1_gaussian_conv` | 8760.8 | 78.1 | 10.41 | 11.457 | 0.1865 |

Against the control: **−29.7 %, −31.4 %, −40.5 %** on the Hessian trace, with a
seed spread of 258.  The BatchNorm gauge accounts for at most 2 % of that, so the
reparametrisation objection does not apply.

**Measured across both BatchNorm policies and both splits** after a companion
loss-landscape study showed that the policy reverses the ranking of the Gaussian
method on *finite-amplitude* sensitivity ([`CROSS_STUDY.md`](CROSS_STUDY.md)):

| tr H vs control | frozen / train | frozen / test | recalibrated / train | recalibrated / test |
|---|---|---|---|---|
| `gaussian_postrelu` | −30.3 % | −42.6 % | −19.2 % | −21.1 % |
| `resolution_max_b1` | −32.0 % | −46.0 % | −28.1 % | −30.5 % |
| `resolution_max_b1_gaussian_conv` | −41.0 % | −54.9 % | −36.9 % | −41.6 % |

**36 measurements out of 36 keep the sign.**  Curvature does not reverse with the
policy, and the effect is *larger* on test images than on training ones, so it is
not a property of the split either.  The policy does change the amount by a
factor of 4.1 to 7.9 — much of what frozen statistics measure is a statistics
mismatch rather than geometry — and the curriculum effect survives that
deflation while shrinking.

The curvature is not uniformly distributed: for the control it runs 248 at the
stem, 687–851 through blocks 0–2, and 2031–3154 in blocks 4–7.  The curriculum
arms lower it everywhere, most in the deep blocks in absolute terms.

---

## 2. The dissociation

This is the result the four frozen methods were designed to make readable, and it
holds.

    gaussian_postrelu   frequency radius  −8.8 %      test error  −16.7 %
    resolution_max_b1   frequency radius  +4.6 %      test error  −17.8 %

**The blur shifts the function's sensitivity towards low frequencies; the
resolution reduction does not shift it at all — slightly the other way — and the
two improve the test error by the same amount.**

A low-frequency bias therefore cannot be the common mechanism.  It is real, and
it is specific to the blur.

What the two interventions *do* share is the **magnitude** of input sensitivity:
‖J‖_F falls 47.8 % and 30.2 %, and the finite-scale response S(3) falls 12.9 %
and 16.5 %.  Not the distribution of sensitivity across frequencies — its size.

---

## 3. The linear regime is smaller than the data's quantisation

The ratio `S(eps) / (eps sigma_max)`, which would be 1 if a first-order
description held:

| eps | 0.05 | 0.25 | 0.5 | 1 | 3 |
|---|---|---|---|---|---|
| `plain` | 0.853 | 0.506 | 0.320 | 0.172 | 0.054 |
| `gaussian_postrelu` | 0.908 | 0.647 | 0.465 | 0.285 | 0.094 |
| `resolution_max_b1` | 0.873 | 0.547 | 0.355 | 0.197 | 0.066 |
| `resolution_max_b1_gaussian_conv` | 0.901 | 0.622 | 0.431 | 0.252 | 0.083 |

**The limit is sound; the window is the problem.**  At `eps = 0.05` the ratio is
0.85-0.91, so a first-order description does hold there, to 10-15 %.  But an
`eps`-norm perturbation spread over `d = 3072` pixels has a per-pixel RMS of
`eps / sqrt(d) = eps / 55.4`, which at `eps = 0.05` is 9.0e-4 -- **0.23 of one
8-bit quantisation step of the image itself**.  The linear regime is confined to
perturbations smaller than the data's own quantisation.  By `eps = 0.5` (two to
three 8-bit levels) a third to a half of the prediction survives, and by
`eps = 3`, a twentieth.

That is the structural reason `sigma_max` and `||J||_F` fail as predictors rather
than merely being coarse: any theory that reasons about neighbourhoods of data
points needs a radius comparable to the spacing between them, and at that scale
the first-order description is already gone.

This has a consequence that shows up as a ranking reversal.  The Gaussian method
leads on the derivative and trails at finite amplitude:

| | ‖J‖_F (derivative) | S(3) (finite) |
|---|---|---|
| `gaussian_postrelu` | **−47.8 %** | −12.9 % |
| `resolution_max_b1` | −30.2 % | **−16.5 %** |

The same reversal appears in weight space between `tr H` and the companion
study's `S(ε)` ([`CROSS_STUDY.md`](CROSS_STUDY.md) §3).  Two independent spaces,
one pattern: an infinitesimal measurement does not predict a finite one here.

---

## 4. No quantity acts as a dial

Twenty conditions, exact gap, partial correlations controlling the training
error, Pearson and Spearman, Bonferroni over the 25 quantities tried, and the
between/within-group decomposition.

| quantity | pooled (n=20) | control (n=5) | curriculum (n=15) | within-group |
|---|---|---|---|---|
| `jac_spectral` | **+0.822** | +0.395 | +0.022 | +0.079 |
| `S@0.05` | **+0.826** | +0.444 | +0.009 | +0.076 |
| `jac_frobenius` | **+0.778** | +0.277 | +0.027 | +0.063 |
| `S@3` | +0.576 | +0.363 | +0.178 | +0.205 |
| `trace_H` | +0.274 | −0.050 | +0.005 | −0.003 |

Every strong pooled coefficient is a **grouping artefact**: the control cells have
both a large predictor and a large target, so the pooled figure is measuring
"is this a curriculum arm", not a relation.  Inside the groups nothing survives —
the within-group pooled coefficients span −0.41 to +0.24, none significant.

The detection ceiling was 1.00 and the accuracy range 12.5 points, so this is not
a power failure.  **The test could answer, and the answer is no.**

### What this retracts

An earlier exploratory study on another branch reported the same finite-scale
response -- `resp_top`, measured the same way -- as a predictor of test error at
+0.72.  On this grid,
with more conditions and a wider range, the same quantity gives **+0.178** inside
the curriculum family.  It does not replicate as a predictor.

It remains a **shared signature**: it falls 12.9 %, 16.5 % and 18.6 % for the
three curriculum methods against the control.  That is a contrast, not a dial,
and it should be described as one.

---

## 5. PAC-Bayes, including the displaced prior

The initialisation-centred bound was already known to be vacuous by a factor of
about 400, with flatness buying 1.65 and the BatchNorm gauge 1.06.

The remaining route was a prior placed at a phase boundary, which needs
‖W* − W_prior‖ divided by 20 to 40.  Measured here from the per-epoch
checkpoints, the distance still to travel at the two boundaries of `Rprog`:

| | epoch 6 | epoch 12 |
|---|---|---|
| `plain` | 31.92 | 16.63 |
| `resolution_max_b1` | 35.92 | 26.07 |
| `gaussian_postrelu` | 19.37 | 14.23 |
| `resolution_max_b1_gaussian_conv` | 22.71 | 13.61 |

Against a total displacement from initialisation of about 13.6.  The network is
**further from its endpoint at the phase boundary than the whole trajectory is
long**.  Placing the prior there makes the bound worse, not better.  That closes
the last PAC-Bayes lever, and it cost no GPU time.

One mechanistic observation from the same measurement: at epoch 1 the blur arms
are already at 28.2 and 30.2 from their endpoint, against 45.1 and 47.7 for the
control and the resolution arm.  **The blur cuts the initial displacement by
40 % while the resolution reduction increases it** — a dissociation in weight
space as well as in frequency, visible from the first epoch.

---

## What this establishes

1. The four methods reproduce the recorded accuracies to within 0.5 pp.
2. Curriculum solutions are flatter by 30–40 % on the Hessian trace, and the
   BatchNorm gauge explains at most 2 % of it.
3. **Blur and resolution reduction reach the same gain by different functional
   mechanisms.**  The frequency shift is specific to the blur.  What they share
   is the magnitude of input sensitivity, not its spectral distribution.
4. A first-order description of the function holds only below `eps = 0.05`, a
   perturbation smaller than one 8-bit quantisation step of the image.
5. No measured quantity predicts test error at equal training error within the
   curriculum family.  Every apparent predictor is a grouping artefact.
6. Every PAC-Bayes route tried is closed, including the displaced prior.

## What it does not

* Any causal claim.  All of it is measured at the endpoint.
* Any statement about seed variance (§0).
* Anything beyond CIFAR-10, ResNet-20 and this protocol.
* The per-epoch ordering of the signatures: the checkpoints exist, the probe is
  written, it was not run.

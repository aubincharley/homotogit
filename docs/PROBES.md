# Probes on a finished solution

Six measurements on the weights a run arrives at, and the guards that decide
whether their numbers can be read.  None of them trains anything.  All of them
evaluate at the **target** intervention state, which at the last epoch is the
exact identity in every method, so all arms are probed on the same network and
the comparison needs no correction.

Run them with `py -m continuation_core {sensitivity,curvature,gauge} <checkpoint>`.

---

## Shared conventions

**Evaluation mode.** `model.eval()` throughout.  With BatchNorm in train mode the
loss depends on the composition of each batch and is not a fixed function of the
weights, so its Hessian is not the object the generalisation bounds refer to.

**Parameter space.** Learnable parameters only.  `running_mean`, `running_var`
and `num_batches_tracked` are buffers: never perturbed, never differentiated,
never counted in a distance.

**Probe set.** The pinned training subset.  A quantity computed on test images
has already seen what it is meant to predict; computed on training images it
remains a predictor.  Nothing here uses labels except the training and test
errors themselves.

**Numerical precision.** Norms and inner products are reduced in float64.  Random
directions are drawn on CPU and moved to the device, so a given seed reproduces
the same perturbation whatever the device.

---

## 1. Curvature — `analysis/curvature.py`

Hessian trace by Hutchinson with Rademacher probes, deflated by the top
eigenpairs; per-block traces free from the same draws; top spectrum by power
iteration with Gram-Schmidt deflation and a **signed** Rayleigh quotient, so a
dominant negative direction would be visible rather than silently made positive.

`H` is 269,722 x 269,722 and is never formed; access is through Hessian-vector
products by double differentiation, accumulated over minibatches with the
correct size weights so the result does not depend on the batch size.

**Cost.** Dominated by the number of Hessian-vector products: `top_k * iters`
plus `draws`.  At the defaults (5 x 40 + 64 = 264) a checkpoint takes about
fifteen minutes on a T4.  At `draws=24, top_k=1, iters=25` (49 products) it takes
three, for a trace precision of about 2.7 % instead of 1.6 % — ample for a
contrast of 30 %, and the right setting unless fine rankings are wanted.

**A measured caution.** Run-to-run noise is 1.7 % on the trace, which is the
estimator's own precision, but **8.2 % on `lambda_max`**.  Fine comparisons on
the top eigenvalue alone sit below that floor; `top_share_of_trace` is stable at
2.2 % and is the statistic to use instead.

## 2. BatchNorm gauge — `analysis/gauge.py`

Every main-path convolution is followed by a BatchNorm, which divides out any
positive per-filter rescaling of the convolution before it.  A fraction of
`||W* - W0||` is therefore pure gauge, and so is part of any curvature measured
in weight coordinates.  The module computes the gauge-minimal distance,

    d_min^2 = sum_f ||W0_f||^2 sin^2(theta_f)   for cos theta_f > 0,

and **measures** the function-equivalence rather than asserting it.  Filters
without a positive optimum are counted, not given the open infimum.  A negative
scale is refused: it is not a symmetry, because BatchNorm divides by a standard
deviation and not by the scale.

Measured elsewhere on this architecture: the gauge accounts for 3 % of the
distance and at most 2 % of the curvature, against effects of 30 to 40 %.  It is
a control that passes, not a correction that rescues anything — but it has to be
run to say so.

## 3. Input-Jacobian spectrum — `analysis/sensitivity.py`

`J = df/dx` in `[0,1]` intensity units with the channel normalisation *inside*
`f`, so the differentiated map is the one the network implements at deployment.

The network has ten outputs, so `J` is `10 x 3072` of rank at most ten and
`J J^T` is a `10 x 10` matrix: the **entire** singular spectrum is exact from the
ten backward passes already needed, with no power iteration and no truncation.

**`||J||_F` is not a generalisation measure.**  It correlates strongly with the
*training* error (−0.59 on the grid of this repository, −0.83 on an earlier one),
so it is largely a proxy for how hard a network fitted its training set.  Any
contrast on it must be read beside the training error.

## 4. Frequency profile — `analysis/sensitivity.py`

The 2-D DFT of each Jacobian row says which input frequencies the logits respond
to.  This is exact and needs no sampling of directions: by Parseval, the sum of
`||J e||^2` over any orthonormal family spanning a radial ring **is** that ring's
DFT energy.

Rings hold very different numbers of modes (roughly `2 pi k`), so the raw ring
sum rises with `k` for a purely geometric reason.  The **per-mode density** is
the quantity that answers "does this function prefer high frequencies?", and both
are returned.

## 5. Amplitude curve — `analysis/sensitivity.py`

    S(eps) = E_x || f(x + eps v_1(x)) - f(x) ||_2

with `v_1(x)` the dominant right singular vector of `J(x)`, recomputed **per
image**: it is a field of directions on the data manifold, not one global
direction.  The amplitude is shared across networks, so the comparison is at
equal perturbation energy.

Reported alongside is the **linearity ratio** `S(eps) / (eps sigma_max)`, which
would be 1 if a first-order description held.  It does not: on this repository's
grid it is already 0.32 to 0.47 at the smallest measurable amplitude and falls to
0.05 by `eps = 3`.  **There is no accessible scale at which the first order
describes this function**, which is why `sigma_max` and `||J||_F` behave as they
do.

## 6. Exact generalisation gap — `tools/job_probe.py`

Training error on the **whole** pinned subset, test error on the whole test set.
Not on a probe: with a 500-image training probe the gap carries about 1.3 points
of sampling noise against a real spread of 0.4, which caps any correlation
against it at 0.25 and makes the analysis unable to answer either way.

---

## The guards — `analysis/power.py`

Four checks, each of which caught a real error.  They are not hygiene.

**1. Validity.**  A probe rebuilds the model and loads a state dict.  An arm
whose trained network carries a *parameter-free* architectural change loads
without error and is then evaluated as a different function.  Metadata does not
catch this — it tracks schedules, not architecture.  The check is empirical: a
cell is admitted only if its probed test error reproduces the one recorded at
training time.  It caught two arms out of twenty-seven elsewhere, one of them
carrying the most extreme value in the set.

**2. Reproducibility floor.**  With no determinism flags, cuDNN's
non-deterministic convolution backward places two otherwise identical runs 6.4 to
7.5 apart in weight space — about half the distance either travels from its
initialisation — while their test accuracies agree to 0.12-0.28 pp.  Signatures
moved only 1-2 % across that, so they are properties of the recipe rather than of
the draw; but rankings closer than twice the floor are not readable, and the
floor must be measured, not assumed.

**3. Detection ceiling.**  A correlation against a noisy target is attenuated.
The ceiling is computed from the target's own noise and printed beside every
coefficient, because a coefficient *at* the ceiling is not a failure — one was
read as one.

**4. Between-group versus within-group.**  A correlation computed across a set
containing two natural groups can be entirely an artefact of the grouping: if one
group has both a high predictor and a high target, the pooled coefficient is
large while the relation inside each group is weak or reversed.

This guard exists because the fix for guard 3 caused the failure it catches:
partialling out the training error *rescued* a quantity on one dataset and
**manufactured** a false positive on this one, +0.83 pooled against +0.02 inside
the curriculum cells.  Both decompositions are printed, always.

The within-group coefficient is pooled across groups by Fisher's z weighted with
each group's degrees of freedom.  Taking a maximum instead lets a small group —
where a coefficient means nothing — speak louder than a large one, which is how
four quantities first slipped past.

### The protocol

* the gap comes from the whole pinned training subset, never a probe;
* correlations are **partial**, controlling the training error;
* **Pearson and Spearman** are always reported together — a single distant point
  manufactured coefficients of +0.94 that fell to +0.44 on ranks;
* correlations are computed **per condition**, not per checkpoint, or the sample
  size is a fiction;
* the Bonferroni threshold uses the number of quantities actually tried;
* a coefficient flagged as a grouping artefact never counts as passing, whatever
  its p-value.

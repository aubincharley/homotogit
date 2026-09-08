# Transformation families beyond the Gaussian baseline

> **Status update.** The TV-budget families of §1 are now **implemented** as
> `tv_l2` and `tv_hminus1` (see [`tv_budget.md`](tv_budget.md)). Section 2's
> operational rate family remains specified but unimplemented. The rest of this
> document is the original specification, kept as the contract those
> implementations had to satisfy.

`continuation/transforms/__init__.py` registers `gaussian`, `tv_l2` and
`tv_hminus1`; asking for `rate` raises a `KeyError` that points here. This file records what each family must satisfy *before* it is
added, so that the eventual implementation cannot quietly skip a requirement.

## 0. The general construction

Given a scalar image-complexity functional $C$, the intended family is

$$T_t^C(x)\in\operatorname*{arg\,min}_{z\in[0,1]^{3\times H\times W}}\tfrac12\|z-x\|_2^2
\quad\text{s.t.}\quad C(z)\le t\,C(x),\qquad t\in[0,1].$$

This picks the **closest admissible reconstruction**. Maximizing $C(z)$ subject
only to $C(z)\le tC(x)$ would be wrong: it does not tether the reconstruction to
the original image and would not even guarantee recovering $x$ at $t=1$.

A new complexity functional must explicitly specify:

1. its **domain**,
2. its **minimum** and what attains it,
3. the **feasible budgets**,
4. the **reconstruction fidelity** criterion,
5. any **tie-breaking rule**.

Nonconvex $C$ need not give a unique or continuous path, and **no family may
assume the Gaussian's semigroup or nesting properties**.

## 1. First planned extension: a TV budget

Channelwise discrete isotropic total variation,

$$\operatorname{TV}(z)=\sum_{c,p}\sqrt{(D_1z_c)_p^2+(D_2z_c)_p^2},$$

with the convention already implemented for *measurement* in
[`continuation/diagnostics.py`](../continuation/diagnostics.py): forward
differences, unit pixel spacing, zero normal difference (Neumann) at the last
row/column of a connected image grid, channels summed after the per-pixel square
root. This TV is a **roughness functional on pixel values**, not a
total-variation distance between probability distributions.

The family is

$$T_t^{\mathrm{TV}}(x)=\operatorname*{arg\,min}_{z\in[0,1]^{3\times H\times W}}\tfrac12\|z-x\|_2^2
\quad\text{s.t.}\quad \operatorname{TV}(z)\le t\operatorname{TV}(x).$$

Properties that follow, and that the implementation must respect:

* the feasible set is closed and convex, and the quadratic fidelity gives a
  **unique** minimizer;
* the exact finite-dimensional solution varies **continuously** with the budget;
* at $t=1$ it is exactly $x$ (the target endpoint is at $t=1$, *not* at 0 — this
  is why `ImageTransform` exposes `target_parameter` instead of assuming zero,
  and why `IncreasingBudgetSchedule` already exists);
* at $t=0$ it is spatially constant per channel, equal to that channel's mean;
* if $\operatorname{TV}(x)=0$, return $x$ at every level.

The $t=0$ endpoint is a mathematical reference, **not** a mandatory starting
point for training: a usable initial budget will have to be chosen
experimentally.

### Penalized route and the calibration trap

The related penalized problem is

$$\operatorname*{arg\,min}_{z\in[0,1]^{3\times H\times W}}\tfrac12\|z-x\|_2^2+\lambda\operatorname{TV}(z).$$

A common $\lambda$ across images and a common relative budget $t$ across images
are **different experimental choices**. Enforcing a particular relative budget
generally requires per-image calibration of $\lambda$. This is why
`ImageTransform.apply` takes a `meta` argument and returns a
`TransformResult.info` dict, and why `cache_key` includes image identity: a
per-image budget needs sample identifiers and image-dependent calibration.

### Reporting requirements for the future solver

Every call must report, in `TransformResult.info`:

* the **requested** budget and, separately, the **achieved** $\operatorname{TV}(z)$;
* the fidelity $\tfrac12\|z-x\|^2$;
* the feasibility error;
* the numerical tolerance and the stopping criterion actually reached;
* the transformation cost (iterations, wall time).

Keys are named `requested_*` and `achieved_*` for exactly this reason. **An
unconverged approximate solve must not claim an exact budget.** Cache keys must
include the solver tolerance.

## 2. Later possibility: an operational rate budget

Shannon rate–distortion,

$$R(D)=\inf_{P_{\widehat X\mid X}:\;\mathbb E[d(X,\widehat X)]\le D} I(X;\widehat X),$$

is a function of a **source distribution**, a distortion criterion and a
reconstruction channel. It is **not** automatically a scalar complexity $C(x)$
of a single image.

An operational implementation must specify a codec or representation, an actual
code-length/rate measure, a decoder, and a distortion criterion. One workable
definition: seek the least-distorted reconstruction achievable within a bit
budget.

Requirements:

* record **actual bits per image / bits per pixel**. A codec *quality setting is
  not a measured rate*;
* handle the fact that a codec family can have **discrete jumps**, a **nonzero
  minimum rate**, and a **lossy maximum-quality setting** — state explicitly
  whether and how the target endpoint recovers the original image;
* do **not** claim a continuous homotopy, or a Shannon-optimal rate–distortion
  solution, merely because a compression parameter was varied.

## 3. Smoothing is not information loss

An ideal Gaussian Fourier multiplier $e^{-\sigma^2|\omega|^2/2}$ is nonzero at
every finite frequency. Without quantization, observation noise or another
irreversible operation, the transformation can remain **injective**. Roughness,
numerical conditioning, coding rate and mutual information are different
notions and must not be equated in any claim this codebase makes.

## 4. Would a simplification even help learning?

Low TV or a low coding rate does **not** imply an easier classification problem:
a transformation can remove exactly the details that distinguish classes. Pixel
fidelity does not guarantee preservation of label-relevant information either.

Usefulness is therefore treated as a **hypothesis to test** — through
optimization behaviour, discriminative performance, and eventual transfer to the
original objective — not as an assumption. The convex, well-posed image
transformation problem must also be kept distinct from the generally nonconvex
network training problem.

`continuation/diagnostics.py` supports the calibration measurements
($\operatorname{TV}(T_\eta x)/\operatorname{TV}(x)$, reconstruction MSE on fixed
subsets, explicit zero-TV handling), with the standing caveat that equal TV or
equal MSE does not mean equal semantic information or equal learning difficulty,
and that Gaussian $\sigma$ is not an exact per-image TV budget.

## 5. What a data-transform interface does *not* cover

Feature-map smoothing, activation homotopies and parameter-space homotopies are
**not** image transformations. They change the model or the objective and need
their own extension points in `continuation/models/` — see the note in that
package's docstring. In particular the feature-smoothing method of *Curriculum
by Smoothing* (Sinha, Garg & Larochelle, NeurIPS 2020) is a different
intervention from the input-only baseline studied here; the two must not be
silently combined or substituted. Learned continuation controllers and learned
representations of the solution path are further changes to the training
procedure, not additional image filters.

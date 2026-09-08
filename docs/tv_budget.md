# TV-budget transformation families: TV-$L^2$ and TV-$\dot H^{-1}$

Implemented in [`continuation/transforms/tv.py`](../continuation/transforms/tv.py),
registered as `tv_l2` and `tv_hminus1`. **Previews only** — these families have
not been used for any training run, and none was launched.

## Common setup

Everything acts on the original float RGB image $x\in[0,1]^{3\times H\times W}$,
*before* network normalization.

Forward spatial differences with unit pixel spacing and **zero difference at the
far boundary** (no periodic wrapping):

$$(D_1z)_{c,p,q}=\begin{cases}z_{c,p+1,q}-z_{c,p,q}&p<H\\ 0&p=H\end{cases}
\qquad\text{and analogously for }D_2\text{ horizontally.}$$

Channelwise isotropic total variation:

$$\operatorname{TV}(z)=\sum_{c,p,q}\sqrt{(D_1z)_{c,p,q}^2+(D_2z)_{c,p,q}^2}.$$

This is the same convention already used for *measurement* in
`continuation/diagnostics.py`, verified equal by
`test_tv_matches_the_measurement_convention_in_diagnostics`.

For $t\in[0,1]$ both methods share the constraint set

$$\mathcal K_t(x)=\Big\{z\in[0,1]^{3\times H\times W}:
\operatorname{TV}(z)\le t\operatorname{TV}(x),\ \bar z_c=\bar x_c\ \forall c\Big\}.$$

The TV budget is **global across channels**; each channel's mean is conserved
**separately**; **no variance conservation** is imposed.

Only the fidelity differs:

| family | fidelity minimized over $\mathcal K_t(x)$ |
|---|---|
| `tv_l2` | $\tfrac12\sum_c\lVert z_c-x_c\rVert_2^2$ |
| `tv_hminus1` | $\tfrac12\sum_c (z_c-x_c)^\top\mathsf L^\dagger(z_c-x_c)$ |

with $\mathsf L=D_1^\top D_1+D_2^\top D_2$ and $\mathsf L^\dagger$ the
Moore–Penrose pseudoinverse, zero on the constant mode. This is the
**homogeneous** $\dot H^{-1}$ form: there is no $\alpha$ and
$(I+\alpha\mathsf L)^{-1}$ appears nowhere in the code.

Because the means are preserved, every residual $r_c=z_c-x_c$ is zero-mean, so
the fidelity equals $\tfrac12\sum_c\langle r_c,p_c\rangle$ with
$\mathsf Lp_c=r_c$, $\bar p_c=0$ — the two formulations are checked against each
other in `test_hminus1_fidelity_two_formulations_agree`.

## The Laplacian and its pseudoinverse

$\mathsf L$ for this boundary convention is the standard Neumann Laplacian, so
it is diagonalized exactly by the DCT-II basis with eigenvalues

$$\lambda_{p,q}=\Big(2-2\cos\tfrac{\pi p}{H}\Big)+\Big(2-2\cos\tfrac{\pi q}{W}\Big).$$

$\mathsf L^\dagger$ is applied as DCT → divide by $\lambda$ (zeroing the constant
mode) → inverse DCT, which is exact rather than iterative.
`test_dct_eigenvalues_match_the_explicit_laplacian_matrix` builds $\mathsf L$
column by column from the difference operators on a 5×4 grid and confirms it is
symmetric, positive semidefinite, and has exactly these eigenvalues.

## Solver

Chambolle–Pock primal–dual (PDHG) applied to

$$\min_z\ f(z-x)+\iota_B(Dz)+\iota_C(z),$$

with $B$ the global isotropic-TV ball of radius $t\operatorname{TV}(x)$ and $C$
the box $[0,1]$ intersected with the per-channel mean hyperplanes. Both
indicators are handled in the dual through $K=[D;I]$, so $\lVert K\rVert^2\le9$
and the step sizes $\tau=\sigma=0.3$ satisfy $\tau\sigma\lVert K\rVert^2=0.81<1$.

* **The relative budget is a hard constraint.** No TV penalty coefficient exists
  anywhere in the implementation. A common fixed $\lambda$ would not be an
  equivalent construction across images, and enforcing a per-image relative
  budget through a penalty would require per-image calibration of $\lambda$;
  projecting onto the TV ball avoids that entirely.
* The TV-ball projection is a **group-$\ell_1$ (ℓ_{1,2}) ball projection**: the
  per-pixel gradient norms are projected onto an $\ell_1$ ball by the standard
  sorting/thresholding algorithm, then each 2-vector is rescaled.
* $\operatorname{prox}_{\tau f}$ is closed-form for **both** fidelities —
  $(v+\tau x)/(1+\tau)$ for $L^2$, and a diagonal multiplication
  $\hat r\mapsto\hat r\,\lambda/(\lambda+\tau)$ in the DCT basis for
  $\dot H^{-1}$ — so the two methods share one solver and differ in exactly one
  function.
* The box+mean projection uses a per-channel bisection on the scalar shift $\mu$
  in $\operatorname{clip}(z+\mu,0,1)$, whose mean is monotone in $\mu$; 50
  vectorized halvings give ~1e-15 accuracy.

**Stopping criterion.** Mean-absolute PDHG primal and dual residuals,

$$r_p=\tfrac1n\big\lVert\tfrac{z_k-z_{k+1}}{\tau}-K^\top(y_k-y_{k+1})\big\rVert_1,
\qquad
r_d=\tfrac1m\big\lVert\tfrac{y_k-y_{k+1}}{\sigma}-K(z_k-z_{k+1})\big\rVert_1,$$

both below `tol` (default $10^{-7}$), checked every 25 iterations, up to
`max_iters` (default **40000**). Tight budgets on natural images need
$\sim10^4$ iterations; an earlier cap of 4000 silently returned infeasible
iterates — one preview solve overshot its budget by 22% — which is why the
default is set for correctness rather than speed.

Since the box and mean constraints are part of $\mathcal K_t$, they are enforced
exactly on the returned iterate by a final projection; the residual TV-budget
violation is then **reported**, never absorbed.

## Endpoints

Handled exactly, without invoking the solver:

* $T_1(x)=x$ (bitwise identity; `target_parameter` is **1**, not 0),
* $T_0(x)=\bar x$, spatially constant per channel and equal to that channel's
  mean, with $\operatorname{TV}=0$,
* $\operatorname{TV}(x)=0\Rightarrow T_t(x)=x$ for every $t$.

## Numerical checks

`tests/test_tv_budget.py`, 28 tests:

* forward differences have zero far-boundary difference and no periodic wrapping;
* $\langle Dz,g\rangle=\langle z,D^\top g\rangle$ to 1e-12 (this caught a real
  sign error in the adjoint during development);
* the TV value agrees with `continuation.diagnostics.total_variation`;
* the DCT eigenvalues match an explicitly assembled $\mathsf L$; $\mathsf L$ is
  symmetric PSD; its kernel is the constants; $\mathsf L^\dagger$ inverts
  $\mathsf L$ on mean-free fields and annihilates the constant mode;
* $\ell_1$-ball, TV-ball and box+mean projections hit their targets;
* endpoints exact, zero-TV images returned unchanged at every $t$;
* at interior budgets: box violation $\le10^{-9}$, mean drift $\le10^{-9}$,
  relative budget violation $<5\times10^{-3}$, achieved ratio $\le t$ + slack;
* achieved TV ratio is monotone in $t$;
* the two fidelities give different solutions, and the $L^2$ solution attains a
  lower $L^2$ fidelity than the $\dot H^{-1}$ solution does (each method is
  optimal for its own criterion);
* the solver is deterministic;
* cache keys separate fidelity, tolerance, parameter and image identity;
* the config signature records that the budget is a hard constraint with no
  penalty coefficient.

## Previews

`scripts/tv_previews.py` (CPU only, no training). It reuses the **same ten
images**, one per class, as the Gaussian `transform_levels.png` — read from the
recorded `results/exp0_gaussian/figures/visualization.json` rather than
reselected — and writes:

* `results/tv_previews/transform_levels_tv_l2.png`
* `results/tv_previews/transform_levels_tv_hminus1.png`
* `results/tv_previews/tv_preview_diagnostics.json`

Columns are $t\in\{1,0.9,0.75,0.5,0.25,0\}$ with a fixed display range $[0,1]$
and no per-image contrast adjustment. The JSON records, per image and per
budget: achieved TV ratio, retained contrast
$\rho=\lVert z-\bar x\rVert_2/\lVert x-\bar x\rVert_2$ (zero denominators marked
`nan` and counted, never given a fabricated value), fidelity value, iterations,
convergence flag, budget/box/mean residuals and solve time.

`rho` is what separates *spatial simplification* from a plain *loss of
contrast*: a method could satisfy a TV budget either by flattening structure
(ρ falls fast) or by reorganising it into fewer, sharper regions (ρ stays high).

```bash
py scripts/tv_previews.py
```

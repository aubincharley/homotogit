# The Gaussian input-smoothing family

## Mathematical setup

For a dataset $\mathcal D=\{(x_i,y_i)\}_{i=1}^n$ with $x_i\in[0,1]^{3\times H\times W}$
and a network $f_\theta$ producing logits, the **target** empirical loss is

$$L_{\mathrm{target}}(\theta)=\frac1n\sum_i \ell_{\mathrm{CE}}(f_\theta(x_i),y_i),$$

and for a transformation family $T_\eta$,

$$L_\eta(\theta)=\frac1n\sum_i \ell_{\mathrm{CE}}(f_\theta(T_\eta x_i),y_i).$$

The Gaussian family is

$$T_\sigma x = G_\sigma * x,\qquad T_0 x = x,$$

with the normalized isotropic Gaussian applied spatially and independently per
colour channel. **$\sigma$ is reported in pixel units.** The native parameter is
$\sigma\ge 0$ and the target endpoint is $\sigma=0$.

In the continuous Fourier transform with angular frequency $\omega$,

$$\widehat{T_\sigma x}(\omega)=e^{-\sigma^2|\omega|^2/2}\,\widehat x(\omega),$$

and with heat time $a=\sigma^2/2$ the multiplier is $e^{-a|\omega|^2}$. The
continuous heat operators satisfy $T_{a_1}T_{a_2}=T_{a_1+a_2}$.

> **This semigroup identity is not claimed for the implementation.** The filter
> here is finite, truncated and reflection-padded. `tests/test_gaussian.py::
> test_discrete_filter_does_not_satisfy_the_semigroup_identity_exactly` measures
> the discrepancy explicitly. The identity motivates the heat-time schedule
> parameterization in `continuation/schedules.py`; it is not a property this code
> relies on.

## Implementation conventions

Implemented in [`continuation/transforms/gaussian.py`](../continuation/transforms/gaussian.py).

| Choice | Value | Rationale |
|---|---|---|
| Separability | 1-D kernel along width then height, `groups=C` depthwise | channels never mix |
| Spatial support | **one fixed** radius $R=\lceil \texttt{truncate}\cdot\sigma_{\max}\rceil$, $K=2R+1$ taps, for *every* $\sigma$ | a per-$\sigma$ rounding rule would add spurious discontinuities along the parameter path |
| Default support | $\sigma_{\max}=3$, `truncate=4` $\Rightarrow R=12$, $K=25$ | covers the whole $\sigma$ grid |
| Normalization | taps $\exp(-d^2/2\sigma^2)$ divided by their sum on the fixed support | discrete kernel sums to 1, so constant images are preserved exactly |
| Padding | `reflect`, $R$ pixels each side | output keeps the input's spatial size |
| $\sigma=0$ | exact passthrough (`y is x`) | no division by zero, no residual blur |
| Value space | float in $[0,1]$, *before* channel normalization | filtered training images are never requantized to 8-bit |
| Out-of-range $\sigma$ | `ValueError` | never silently retruncate the kernel |

Requesting $\sigma>\sigma_{\max}$ is an error rather than a silent change of
support, and $R\ge H$ or $R\ge W$ is an error because PyTorch's reflect padding
requires the pad to be smaller than the dimension.

### Continuity at the target endpoint

$\sigma=0$ is special-cased for numerical reasons, but the family is *continuous*
there: as $\sigma\to0^+$ the normalized taps tend to a discrete delta. In float32
they underflow to an exact delta well before $\sigma$ reaches 0, so
$\|T_\sigma x - x\|_\infty$ decreases monotonically to exactly 0
(`test_kernel_tends_to_delta_as_sigma_goes_to_zero`).

### Boundary effects

Reflection padding is exact for constant images, but within $R$ pixels of an
edge the image is smoothed against its own mirror image rather than against
unseen content. With $R=12$ on 32×32 CIFAR images this affects most of the
frame at large $\sigma$, so it is documented rather than hidden:

* `figures/boundary_effects.png` (from `visualize`) plots centre-row profiles of
  a constant image and a vertical step at each level;
* `test_boundary_effect_is_bounded_and_documented` pins the qualitative
  behaviour: flat far from the step, monotone transition, no ringing (the kernel
  is non-negative);
* the attenuation figure and test measure amplitudes on the **interior** half of
  the image to keep the border out of the frequency measurement.

## What was verified

`tests/test_gaussian.py`, all passing:

* exact identity at $\sigma=0$ (bitwise, and the same object);
* kernel normalization, non-negativity and symmetry; delta at $\sigma=0$;
* continuity of the family as $\sigma\to0^+$;
* constant images preserved at every level (values 0, 0.5, 1);
* shape, dtype and value-range preservation (output is a convex combination of
  inputs, so it stays in $[0,1]$);
* determinism across calls and across separate instances;
* $T_\sigma(T_\sigma x)\neq T_\sigma x$ — a reminder that the trainer always
  transforms the **original** image, never a previously transformed one;
* channel independence (a delta in one channel leaves the others exactly zero,
  and its mass is conserved);
* the separable implementation equals an explicit dense 2-D convolution to 1e-9;
* attenuation of sinusoidal gratings decreases monotonically in both frequency
  and $\sigma$, and matches $e^{-\sigma^2\omega^2/2}$ to 2 % at low frequency
  (high frequencies are *not* required to match — see the figure);
* fixed support independent of $\sigma$; errors for $\sigma>\sigma_{\max}$,
  $\sigma<0$, and a radius too large for the image;
* heat-time helpers round-trip;
* the discrete filter does **not** satisfy the semigroup identity exactly;
* cache keys separate image identity, family configuration and parameter value.

## Deliberately not implemented

Hard Fourier cutoffs and frequency projections. The family is a smooth
multiplier, and an ideal Gaussian multiplier is nonzero at every finite
frequency — so, absent quantization or added noise, the transformation can
remain injective. Visual smoothing is **not** the same thing as a strict
decrease in Shannon information; see [`extensions.md`](extensions.md).

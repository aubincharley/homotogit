# Differentiable wavelet shrinkage

Core operation: [`continuation/transforms/wavelet.py`](../continuation/transforms/wavelet.py),
`wavelet_shrink(x, s, wavelet, levels=2)`, a plain PyTorch tensor function on
real `[B,C,H,W]` tensors — images **or** activations — CPU/GPU, fully
differentiable. Registered as the `wavelet` family. **Previews and benchmarks
only: no training was launched and the training architecture is unchanged.**

PyWavelets is used solely to fetch filter coefficients and as a test reference;
it is never on the compute path.

## Construction

**Boundary.** Mirror the input to `2H x 2W` by concatenating with its reversal
along each axis, run the *periodic* stationary transform there, crop the
top-left `H x W` block. Opposite edges of the original image are never
periodically joined.

**Transform.** Two-level, separable, undecimated (à trous), per sample and
channel; level `j` dilates filters by `2^(j-1)`. Each 1-D filter is scaled by
`1/sqrt(2)`, making the analysis a tight frame with bound one (`W*W = I`);
synthesis is the exact adjoint `W*`. Retained: `a_2` plus all six detail bands.

Orientation: first letter = row axis, second = column axis (`LH` = lowpass rows,
highpass columns). All three detail orientations are thresholded identically, so
this affects labels only.

**Shrinkage.** Per sample, channel and band:
`nu = ||d||_2/sqrt(N)`, `lambda_{j,o}(s) = 4(1-s) 2^(1-j) nu`,
`T_s = P W* (a_2, soft(d, lambda))`. `a_2` untouched; `nu` from unthresholded
coefficients, never pooled across samples/channels, differentiable, with
`nu = sqrt(mean(d^2)+eps)` giving a finite (zero) gradient on zero bands.

No identity shortcut: `T_1 = P W* W E = I` is produced by the actual round-trip
(~3e-7 in float32, 1e-12 in float64). `T_0` is **not** constant. No clipping,
normalization, or TV/mean constraint is applied; outputs may leave `[0,1]`.

This is an explicitly defined shrinkage operator — not a constrained-TV solution
and not a claimed proximal operator for redundant wavelet analysis.

## Verification (`tests/test_wavelet.py`, 43 tests)

* **Round-trip** `W*W = I` to 1e-12, tested directly on `swt2_analysis` /
  `swt2_synthesis`, so no identity shortcut could satisfy it.
* **Adjoint** `<Wu, v> = <u, W*v>`; **Parseval** energy identity to 1e-12.
* **PyWavelets reference**: per-band energies match `pywt.swt2(norm=True)` to
  1e-13 for all four wavelets. Our alignment differs from PyWavelets by a
  per-band circular shift (a shift of a Parseval frame is the same frame, and
  the shrinkage rule depends only on shift-invariant band RMS), so energies —
  not coefficient arrays — are the meaningful comparison.
* **Constant inputs**: all detail bands vanish (<1e-12) and `T_s(const) = const`
  for every `s`.
* **Gradcheck** in float64 away from threshold boundaries; gradients flow
  through the RMS thresholds; zero band gives a finite zero gradient.
* RMS is per-sample/channel: rescaling one sample leaves the others bit-identical
  and rescales its own output proportionally.
* Filter support that does not fit the extended axis raises rather than wrapping.

## Preview diagnostics (10 images, CPU)

Mean over the ten images; full per-image data in
`results/wavelet_previews/wavelet_preview_diagnostics.json`.

| s | TV ratio (haar/db2/sym4/coif1) | rho | zeroed details |
|---|---|---|---|
| 1 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 | 0.000 |
| 0.9 | 0.799 / 0.825 / 0.833 / 0.827 | 0.948–0.962 | 0.41 |
| 0.75 | 0.617 / 0.661 / 0.678 / 0.663 | 0.890–0.922 | 0.66 |
| 0.5 | 0.472 / 0.525 / 0.552 / 0.527 | 0.831–0.885 | 0.85 |
| 0.25 | 0.415 / 0.469 / 0.500 / 0.472 | 0.801–0.868 | 0.93 |
| 0 | 0.389 / 0.445 / 0.477 / 0.448 | 0.785–0.859 | 0.97 |

* **`rho` uses each image's own mean** (`||z-zbar||/||x-xbar||`), as specified
  here; the TV previews used the original's mean. `retained_contrast(center=…)`
  now supports both. Zero denominators are marked `nan` and counted.
* **Channel mean drift**: haar ~1e-15 (exact); db2/sym4/coif1 ~1e-4. The longer
  filters leak a little DC across the crop boundary.
* **Out of `[0,1]`**: haar stays inside; db2 reaches `[-0.003, 1.040]`, sym4
  `[-0.041, 1.059]`, coif1 `[-0.005, 1.036]` at `s=0`. Raw outputs are kept —
  clamping happens only inside the figures, which use fixed `[0,1]` limits.
* The effect **saturates**: most of the change occurs by `s≈0.5`, and `s=0.25 →
  0` moves TV ratio by only ~0.03. `s` is not close to linear in visual effect.

## Cost vs Gaussian (`results/wavelet_previews/wavelet_timings.json`)

Wavelet at `s=0.5` (complete operation: extension, analysis, RMS, threshold,
synthesis, crop) against Gaussian at `sigma=1`, matched device/dtype/shape,
medians after warm-up, CUDA-synchronised, transfers and I/O excluded, no
optimizer steps. Two Gaussian configs are shown: `sigma_max=3` is the
Experiment-0 setting (radius 12), `sigma_max=1` the smallest support covering
`sigma=1` (radius 4) and the only one that fits 8×8 maps.

**CPU, single image `[1,3,32,32]`, forward (ms):** gaussian 0.43 (either config);
haar ~6, db2 ~14, coif1 15.0, sym4 15.5 — i.e. **14–36× the Gaussian**.

**GPU (RTX 3050), forward / forward+backward (ms), peak allocated (MB):**

| shape | gaussian σmax1 | haar | db2 | sym4 | coif1 |
|---|---|---|---|---|---|
| input `[128,3,32,32]` | 0.31 / 0.76 / 11.6 | 9.5 / 76.0 / 378 | 22.7 / 56.8 / 388 | 25.9 / 61.6 / 403 | 24.5 / 59.1 / 394 |
| stage1 `[128,16,32,32]` | 1.28 / 2.66 / 62 | 46.4 / 235.9 / 2030 | 117.7 / 292.4 / 2072 | 134.3 / 319.3 / 2144 | 126.8 / 306.8 / 2090 |
| stage2 `[128,32,16,16]` | 0.72 / 1.54 / 34.5 | 24.6 / 203.9 / 1019 | 60.7 / 152.5 / 1060 | 69.7 / 170.6 / 1130 | 65.5 / 161.0 / 1091 |
| stage3 `[128,64,8,8]` | 0.45 / 0.98 / 20.0 | 13.1 / 65.0 / 518 | 31.4 / 81.5 / 560 | 36.9 / 95.2 / 633 | 34.2 / 88.3 / 590 |

`gaussian sigma_max=3` is 1.6–2.2× the `sigma_max=1` config and is **n/a** on
8×8 maps (radius 12 exceeds the spatial size).

## Limitations

1. **~50–100× slower than Gaussian on GPU** and 14–36× on CPU. The undecimated
   transform works on a 4× larger domain and materialises ~24 filtered tensors
   per call.
2. **Memory is the binding constraint**: 2.1 GB peak for forward+backward on
   `[128,16,32,32]`, over half of this 4 GB card. Applying it to feature maps
   during training would need a smaller batch or a fused implementation. Both
   are avoidable by construction (fusing the circular pad into the convolution,
   or a custom autograd function), which was out of scope here.
3. **Haar's backward is anomalously slow** — 76 ms vs db2's 57 ms despite a much
   faster forward (9.5 vs 22.7 ms), stable across three independent re-runs. The
   likely cause is the circular-pad backward dominating when the kernel is only
   2 taps, but I have not confirmed this.
4. Alignment differs from PyWavelets by a per-band circular shift (see above);
   energies match, coefficient arrays do not.
5. Non-haar wavelets have ~1e-4 channel mean drift and leave `[0,1]` slightly.

## Reproducing

```bash
py -m pytest tests/test_wavelet.py -q
py scripts/wavelet_previews.py
py scripts/wavelet_benchmark.py
```

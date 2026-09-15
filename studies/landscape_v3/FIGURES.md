# Figure index (landscape_v3)

All figures are in `figures/` as PDF and PNG, produced by
`py -m landscape_v3.analyze`. The numbers behind them are in `tables/`. The v2
figures in `studies/landscape_v2/figures/` are unchanged and remain valid for
what they show.

## Common conventions

- **Networks.** The 20 v2 networks: plain, resolution-only, Gaussian-only and
  combined, each at seeds 0–4. Seeds are the replication unit; dots are seeds.
- **Centre CE.** The unperturbed mean cross-entropy (CE) on the stated data
  under the stated BatchNorm policy.
- **Policies.**
  - *saved/frozen*: the checkpoint's running statistics;
  - *recalibrated at every point*: 2,000 training images, recalibrated for
    every evaluated weight vector;
  - *recalibrated at centre, then frozen*: calibrated once at the unperturbed
    weights.
- **ε and t.** Both are RMS relative block displacement r over the 698 conv
  filters and fc rows.
  - Random directions are Li et al. filter-normalised, so r = ε.
  - Hessian eigendirections are mapped to weight space and scaled to r = 1
    with one scalar.
- **Surfaces.** Every vertex is a measured point, and missing vertices are
  grey crosses. Absolute CE uses viridis; centre-subtracted CE and paired
  differences use a diverging colour map centred at 0.

## Reading order

### 1. Final solutions across amplitudes (Q1)

| file | content |
|---|---|
| `A1_centre_ce` | centre CE of the epoch-30 networks under the three policies |
| `A2_sensitivity_vs_amplitude` | S̄(ε) for ε = 0.005–0.5, 20 directions, both probes, three policies |
| `A3_paired_difference_vs_amplitude` | Δ = S̄(method) − S̄(plain) within each seed |
| `A4_curvature_proxy` | C̄(ε) = mean 2S/ε², with the HVP quadratic form mean dᵀHd for the two frozen policies |
| `A5_direction_spread` | per-seed spread over the 20 directions |
| `A6_float64_check` | float32 vs float64 S at small ε |

### 2. Larger evaluation sets (Q2)

| file | content |
|---|---|
| `B1_validation_probe_vs_large` | per-seed Δ on the probe vs on the 10,000-image training subset and the full test set; identical 20 directions |

### 3. During training (Q3)

| file | content |
|---|---|
| `C1_temporal_train_probe`, `C1_temporal_test_probe` | centre CE and S̄(0.1), S̄(0.25) at epochs 3–30, produced vs final state, two policies |
| `C2_weight_norms` | weight norms at the same checkpoints |
| `C3_temporal_paired` | paired differences against plain at each epoch |
| `C4_transitions` | fixed weights at every schedule transition: before / after / final state |

### 4. Hessian (Q4)

| file | content |
|---|---|
| `D1_hessian_top1`, `D1_hessian_top2`, `D1_hessian_min` | extreme eigenvalues, both coordinates, both frozen policies, both probes |
| `D2_lanczos_convergence` | residual histories of all eigenproblems and both starts |
| `D3_eigendirection_finite_difference` | float64 finite differences vs the quadratic form |
| `D4_eigendirection_layer_energy` | where the extreme eigendirections put their relative displacement |
| `D5_cuts_<coords>_<policy>` | measured 1D cuts along the eigendirections |
| `D6_cuts_frozen_vs_pointwise` | the same directions evaluated with pointwise recalibration |

### 5. Surfaces

| file | content |
|---|---|
| `E_wide_random_seed0_g41_*` | random plane 0/1 over [−0.5, 0.5]², seed 0, 41×41: absolute / centred, 3D / contour, per probe |
| `E_wide_random_allseeds_g21_*` | the same for all seeds, 21×21 |
| `E_hessian_plane_<policy>_<v1>_<v2>_seed0_*` | seed-0 planes of relative eigendirections with measured cuts |
| `surfaces_interactive.html` | rotatable seed-0 test-probe surfaces |

### 6. Interpolation and PCA (tables only; v2 figures `D1` and `T1`–`T5`)

| table | content |
|---|---|
| `F_interpolation_barriers.csv` | barriers re-summarised from v2 raw data |
| `F_barrier_comparison_res_comb_vs_plain_res.csv` | corrects the v2 blanket claim |
| `F_pca_projection_quality.csv` | explained variance and reconstruction errors |

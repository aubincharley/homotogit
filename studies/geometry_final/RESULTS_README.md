# geometry_results.zip

All tables use CIFAR-10, ResNet-20-BN, SGD and the final 30-epoch checkpoints, with training seeds 0–4. Methods are
`plain`, `resolution_max_b1` (R), `gaussian_postrelu` (G) and `resolution_max_b1_gaussian_conv` (RG).

## New computations (this study)

| file | content |
|---|---|
| `tables/T_trace_per_checkpoint.csv` | 40 rows (method × seed × probe), centre-frozen BN, 268,336 conv + fc weights. For each of tr(H), tr(AᵀHA), tr(HC) (columns `trace_ordinary`, `trace_relative`, `trace_covariance`): estimate, Monte Carlo SD / SEM / relative SEM over 128 draws, and the control-variate estimate with its SEM. Also the existing λ (Lanczos), loss, gradient norm, r-unit scales, and the existing 20-direction quadratic-form mean with the z-difference to tr(HC). |
| `tables/T_trace_draws.csv` | 5,120 per-draw aggregates, with Rademacher seed, z sha256 and seconds |
| `tables/T_trace_draws_control_variate.csv` | per-draw control-variate aggregates (existing top1/top2 ordinary eigenpairs) |
| `tables/T_trace_q_blocks.npz` | per-draw per-block q_g (698 blocks), key = objective, plus `<objective>__draws` |
| `tables/T_trace_paired_vs_plain.csv` | per seed and probe: paired method − Plain difference, Monte Carlo SEM of paired draws, z, resolved flag (\|z\| > 2), ratio and delta-method SEM; also eigenvalue and 20-direction ratios |
| `tables/T_trace_seed_summary.csv` | per method, probe and quantity: mean and SD over seeds, mean MC SEM, ratio to Plain, seeds below / above (resolved) / unresolved |
| `tables/T_rank_orders.csv`, `tables/T_lambda_trace_agreement.csv` | within-seed orderings; whether λ and trace agree in sign against Plain |
| `tables/T_surface_vertices_seed0.csv` | 13,448 rows: 4 models × 41 × 41 × {pointwise, centre_frozen}. Raw and centre-subtracted CE and accuracy on both probes, with the source of every value (`v3:` / `v2:` reused, `geometry_final:` new) |
| `raw/main_job/` | Kaggle outputs: `trace/*.jsonl` (per draw), `trace/*.meta.json` (identity and state checks), `trace/blocks__*.npz`, `grid/*.jsonl` (per vertex), `checks/` (reuse reproduction, precision after 64 and 128 draws, input digests), `environment.json`, `timing.json`, `manifest.json`, `COMPLETE.json` |
| `raw/timing_pilot/` | T4 timing pilot |
| `checks/local/dense_reference.json` | dense-reference validation of the estimator |

## Re-analyses of existing measurements (no new evaluation)

| file | content |
|---|---|
| `tables/X1_bn_policy_finite_*` | identical random perturbations under centre-frozen, pointwise and saved BN; ratios and paired changes against Plain |
| `tables/X2_bn_policy_eigencuts.csv` | leading and min eigendirection cuts under centre-frozen and pointwise BN; raw CE and rise |
| `tables/X3_direction_curvature_per_checkpoint.csv` | extreme eigenvalues vs the 20 random quadratic forms, in ordinary and r units |
| `tables/X4_amplitude_random.csv`, `tables/X4_amplitude_eigen.csv` | finite difference ÷ HVP quadratic form by amplitude |
| `tables/X5_eigvec_radial_fraction.csv` | radial (filter-rescaling) share of every stored extreme eigenvector |
| `tables/T_sensitivity_paired.csv` | paired ΔS behind the regenerated sensitivity figures |
| `tables/T_hessian_plane_vertices_seed0_centre_frozen.csv` | top1/top2 plane vertices with u, t and extents |

Units: CE in nats; curvature per squared parameter unit (ordinary) or per unit r² (r = RMS relative block displacement).

# landscape_v2 results bundle

Four frozen methods x 5 paired training seeds (0-4), CIFAR-10 / ResNet-20, 30 epochs.
Choices fixed beforehand: preregistration.json. Interpretation: REPORT.md.

Common columns
- method: plain | resolution_max_b1 | gaussian_postrelu | resolution_max_b1_gaussian_conv
- seed: training seed (replication unit, n=5). direction: random-direction index within a seed
  (matched across methods of the same seed; not independent replications).
- policy: saved (checkpoint BatchNorm statistics) | recalibrated (BN statistics re-estimated on a fixed
  2,000-image train calibration set; calibration_10k = 10,000 images).
- *_ce / *_acc: mean cross-entropy (no weight decay) / accuracy on
  train_probe, test_probe (1,000 class-balanced images each), train_large (10,000 train),
  test_full (10,000), train_full (50,000). Empty = not evaluated for that row.
- amplitude, sign: relative filter-wise perturbation theta + sign*amplitude*d (Li et al.); amplitude 0 = centre.
  Symmetric sensitivity S = (L(+) + L(-))/2 - L(centre), computed within (method, seed, policy, direction).
- final state = full resolution, no filter (identical to plain ResNet-20).

Files (sensitivity_evaluations.csv: 8040 rows, validation_evaluations.csv: 396 rows, interpolation_curves.csv: 1020 rows, fixed_weight_evaluations.csv: 8160 rows, surfaces.csv: 22600 rows, checkpoint_losses.csv: 3640 rows, training_curves.csv: 620 rows, pca_checkpoints.csv: 1100 rows, pca_variance.csv: 5 rows, pca_plane_grid.csv: 961 rows)
- sensitivity_evaluations.csv   primary: epoch-30 solutions, 20 directions, 5 amplitudes, both signs, both policies, final state
- validation_evaluations.csv    prespecified points re-evaluated on larger sets (check=larger_sets) and with 10k calibration (check=calibration_10k, seed 0)
- interpolation_curves.csv      straight segments between final solutions of a seed, 51 alphas (A at 0, B at 1), recalibrated
- fixed_weight_evaluations.csv  checkpoints at each schedule's first/last transition update, evaluated under several intervention states (resolution, sigma_G); 10 directions, recalibrated perturbations + centres under both policies
- surfaces.csv                  2-D grids along directions 0 (a) and 1 (b), recalibrated; kind=final (all seeds 21x21, seed 0 also 41x41) or fixed_weight (seed 0, 21x21)
- checkpoint_losses.csv         every saved checkpoint (epochs + transition windows) under final and current state, both policies
- training_curves.csv           records written during training (saved BN; pinned 500-image train probe, full test set; current and target state)
- pca_checkpoints.csv           per-seed shared PCA (fit on epoch checkpoints of the 4 methods): PC1-10 coordinates, residual norms after 2/5/10 PCs
- pca_variance.csv              explained-variance ratios per seed
- pca_plane_grid.csv            seed 0: final-state loss of points reconstructed in the PC1-PC2 plane (not the checkpoints' losses)
- integrity.json                per account: timings, download hash verification, checks, pairing digests, task statuses, run summaries
- S2_paired_differences.png, D1_interpolation_segments.png  two key figures

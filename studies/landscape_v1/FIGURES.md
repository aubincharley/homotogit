# Figure manifest (landscape study v1)

All files are in `figures/`, each as PDF and PNG. The source modules are:

- `py -m landscape_study.figures`: contours, slices and segments;
- `py -m landscape_study.surfaces`: 3D surfaces, the checkpoint CE plot and the
  HTML view.

Raw values are in `results/`. The paper was not edited.

**Common to every figure.**

- **Runs:** plain vs resolution-only, resbench checkpoints, not the unified
  reference runs.
- **BatchNorm:** recalibrated on fixed calibration images unless stated.
- **Metrics:** measured on probes of 1,000 class-balanced images.
- **Grid rendering:** every surface vertex is a measured grid point.
- **Missing methods:** Gaussian-only and combined are not shown, because no
  checkpoints exist.

## Suggested reading order

### 1. Final surfaces and robustness slices

| file | caption |
|---|---|
| `A_surface3d_absolute` | CE around the epoch-30 plain and resolution-only solutions (seed 0, target state), train and test probes. The two methods use matched relative filter-wise perturbations, not an identical physical plane. Axes, camera, vertical limits and colour scale are shared within each split. |
| `A_surface3d_centered` | The same grids minus each surface's centre CE, separating shape from loss level. |
| `A_local_landscapes_seed0` | Contour companion of both A surface figures. |
| `A_robustness_1d_slices` | CE increase along six filter-normalized directions per epoch-30 solution, seeds 0–2. At a relative step of 0.25, resolution-only increases less in 18 of 18 training-probe comparisons and 15 of 18 test-probe comparisons (6 directions × 3 seeds, not 18 independent runs). |

### 2. Fixed-weight surfaces across resolution states

| file | caption |
|---|---|
| `B_surface3d_absolute` | One resolution-only checkpoint (seed 0, after epoch 6, trained last at r = 16) evaluated at r = 16, 24 and 32. Weights and directions are fixed across states. The loss level rises with the state. |
| `B_surface3d_centered` | Same grids minus each centre, with negative values kept. The r = 16 and r = 24 slices look alike; at r = 32 the centre is not the lowest sampled point of the slice. |
| `B_fixed_weights_resolution_states` | Contour companion of both B surface figures. |
| `B_secondary_1d` | 1D slices: resolution-only epoch 12 under three states, and plain epoch-6 weights run with the reduction hook. Each set of weights has its lowest loss under the state it was trained in. |

### 3. Straight-line interpolation

| file | caption |
|---|---|
| `D_interpolation` | CE on straight segments between paired epoch-30 plain (α = 0) and resolution-only (α = 1) solutions, seeds 0–2, 51 points. The grid-estimated barrier above the higher endpoint is 0.80–0.87 nats on the training probe and 0.43–0.56 on the test probe. It concerns straight segments only and does not establish disconnected basins. |

### 4. PCA, with its sampling and projection limitations

| file | caption |
|---|---|
| `C_pca_trajectories` | Main view. Projections of five checkpoints per run (seed 0; epochs 0, 6, 12, 18, 30) onto the shared top-two principal directions: 91% of the variance of the 10 points, but relative residual up to 0.66 at epoch 6. Lines join sampled checkpoints and are not observed paths. The background is the target loss of reconstructed in-plane points. |
| `C_surface3d_plane` | The same evaluated 25×25 plane grid as a translucent surface. Projected checkpoints appear only on the floor, never at a height; the loss of the off-plane checkpoints is not this surface. |
| `C_checkpoint_ce` | Actual checkpoint CE by BatchNorm convention: recalibrated vs saved statistics, target vs own state. Early resolution-only target-path degradation is much larger with saved statistics, and it persists after recalibration. |

## Interactive view

`surfaces_interactive.html` contains the A and B grids as rotatable surfaces
(plotly, self-contained) with shared ranges per comparison block. The static
figures above stand on their own.

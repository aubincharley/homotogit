# Geometry figures

Every figure is a vector PDF with a PNG preview. Captions are standalone; `source` names the table or raw file each figure is drawn from (inside `geometry_results.zip`). Regenerate with `py -m geometry_final.figures` and `py -m geometry_final.curvature_figs`.

Suggested use: main text `F_surfaces_centre_frozen_seed0_zoom`, `F_curvature_ratios_to_plain`, `F_hessian_vs_random_rise_seed0`, `F_sensitivity_centre_frozen_pointwise`; everything else appendix.

## F_curvature_ratios_to_plain

Within-seed ratio of each Hessian quantity to Plain: CIFAR-10 / ResNet-20-BN / SGD, final 30-epoch checkpoints, five training seeds (dots; bar = mean), 1k-image training (top) and test (bottom) probes. Mean cross-entropy without weight decay; centre-frozen BatchNorm (statistics recalibrated once at the unperturbed weights on 2,000 training images, then held fixed); variables are the 268,336 convolution and classifier weights. Eigenvalues: existing Lanczos estimates (converged, two starts). Traces: Hutchinson estimates with 128 Rademacher draws shared across methods within a seed; vertical bars are $\pm1$ Monte Carlo SEM of the ratio (delta method, paired draws). $A_g=\|\theta_g\|I$ per filter or classifier row; $C_g=\|\theta_g\|^2/p_g\,I$ is the covariance of the filter-normalised random directions, so $\mathrm{tr}(HC)$ is the mean curvature those directions see. Traces are signed; negative eigenvalues exist.

Source: `tables/T_trace_paired_vs_plain.csv`

## F_curvature_scales_r_units

Three averages of the same centre-frozen Hessian on one common scale, curvature per unit squared RMS relative block displacement $r$ (five seeds per method, log axis): the steepest direction ($G\lambda_{\max}$ of the block-relative Hessian, $G=698$ blocks), the average over directions isotropic in block-relative coordinates, and the average under our filter-normalised random-direction distribution. The steepest direction is about $10^4$ times steeper than the average random direction. CIFAR-10 / ResNet-20-BN / SGD final checkpoints, 1k-image probes.

Source: `tables/T_trace_per_checkpoint.csv`

## F_hessian_plane_top1_top2_centre_frozen_seed0

Existing seed-0 planes spanned by each model's own two leading eigenvectors of the centre-frozen, block-relative Hessian (1k-image training probe), mapped to weight space ($\delta=Aq$) and scaled to $r=1$; the eigenvectors differ between models and are not orthogonal in weight space. Vertices $u_1,u_2\in[-1,1]$ (41-by-41) are displaced by $t_i=u_iT_i$ with common extents $T_1=0.0012$ and $T_2=0.0015$ in $r$ units, fixed from Plain's eigenvalues so that Plain's quadratic model rises by 2 nats at each axis end. These extents are about 20 times smaller than one step of the random grid (0.025). Heights are measured CE above centre; identical camera, limits and colour scale (0 to 5.56 nats).

Source: `landscape_v3 raw hplane__*_train_centre_frozen_rel__top1_top2.jsonl`, `tables/T_hessian_plane_vertices_seed0_centre_frozen.csv`

## F_hessian_vs_random_rise_seed0

Measured symmetric rise $S(r)=\tfrac12[L(\theta+r\delta)+L(\theta-r\delta)]-L(\theta)$ on the 1k-image training probe for the seed-0 final models, with displacement expressed as RMS relative block displacement $r$ (so random and Hessian directions share one axis). Red: the leading eigenvector of the block-relative centre-frozen Hessian (each model's own; not a shared direction), scaled to $r=1$. Colour: 20 filter-normalised random directions (mean, band min--max over directions, centre-frozen). Solid: centre-frozen BatchNorm; dashed: the same perturbations with pointwise recalibration. Dotted: $\tfrac12 G\lambda_{\rm rel}r^2$ and $\tfrac12\overline{d^\top Hd}\,r^2$ from the HVP measurements. Log--log axes; non-positive $S$ omitted. Existing measurements only.

Source: `landscape_v3 tables/D_hessian_cuts.csv`, `landscape_v3 tables/A_sensitivity_per_direction.csv`, `tables/X3_direction_curvature_per_checkpoint.csv`

## F_sensitivity_centre_frozen_pointwise

Finite sensitivity relative to Plain, centre-frozen (left pair) and pointwise-recalibrated (right pair) BatchNorm: CIFAR-10 / ResNet-20-BN / SGD, final 30-epoch checkpoints, 1k-image probes. For each seed, $S$ averages 20 filter-normalised directions (both signs) and $\Delta S=S_{\rm method}-S_{\rm Plain}$; points are the mean of five seed-level differences and bands $\pm1$ sample SD across seeds. Negative means a smaller CE increase. Logarithmic amplitude axis; vertical scales differ between panels. Existing measurements; no new evaluation.

Source: `landscape_v3 tables/A_sensitivity_per_seed.csv`, `tables/T_sensitivity_paired.csv`

## F_sensitivity_saved_appendix

Finite sensitivity relative to Plain, saved BatchNorm statistics: CIFAR-10 / ResNet-20-BN / SGD, final 30-epoch checkpoints, 1k-image probes. For each seed, $S$ averages 20 filter-normalised directions (both signs) and $\Delta S=S_{\rm method}-S_{\rm Plain}$; points are the mean of five seed-level differences and bands $\pm1$ sample SD across seeds. Negative means a smaller CE increase. Logarithmic amplitude axis; vertical scales differ between panels. Existing measurements; no new evaluation.

Source: `landscape_v3 tables/A_sensitivity_per_seed.csv`, `tables/T_sensitivity_paired.csv`

## F_surface_axis_profiles_seed0

Measured CE above centre along the two axes of the seed-0 41-by-41 grid (each axis is one filter-normalised random direction), centre-frozen (solid) and pointwise-recalibrated (dashed) BatchNorm, CIFAR-10 / ResNet-20-BN / SGD final checkpoints, 1k-image probes. Logarithmic vertical axis; the centre and 18 non-positive values (including the centre itself) cannot be drawn on it and are omitted from the lines only.

Source: `tables/T_surface_vertices_seed0.csv`

## F_surfaces_centre_frozen_seed0

Loss slices of the four final seed-0 models under centre-frozen BatchNorm (statistics recalibrated once at the unperturbed weights on 2,000 training images, then held fixed): CIFAR-10 / ResNet-20-BN / SGD, 30 epochs, 1k-image training probe. Surfaces join the 41-by-41 measured vertices of $L(\theta_0+a d_1+b d_2)-L(\theta_0)$ over $[-0.5,0.5]^2$; no smoothing or fitting. Viewing angle, axis limits and colour scale are identical across panels (range 0 to 171 nats). The random draws are shared, but each model scales them by its own filter norms, so the four panels do not show one physical plane. One seed and one direction pair: an illustration, not a ranking.

Source: `tables/T_surface_vertices_seed0.csv`

## F_surfaces_centre_frozen_seed0_test

Loss slices of the four final seed-0 models under centre-frozen BatchNorm (statistics recalibrated once at the unperturbed weights on 2,000 training images, then held fixed): CIFAR-10 / ResNet-20-BN / SGD, 30 epochs, 1k-image test probe. Surfaces join the 41-by-41 measured vertices of $L(\theta_0+a d_1+b d_2)-L(\theta_0)$ over $[-0.5,0.5]^2$; no smoothing or fitting. Viewing angle, axis limits and colour scale are identical across panels (range 0 to 171 nats). The random draws are shared, but each model scales them by its own filter norms, so the four panels do not show one physical plane. One seed and one direction pair: an illustration, not a ranking.

Source: `tables/T_surface_vertices_seed0.csv`

## F_surfaces_centre_frozen_seed0_zoom

Loss slices of the four final seed-0 models under centre-frozen BatchNorm (statistics recalibrated once at the unperturbed weights on 2,000 training images, then held fixed): CIFAR-10 / ResNet-20-BN / SGD, 30 epochs, 1k-image training probe. Surfaces join the 9-by-9 measured vertices of $L(\theta_0+a d_1+b d_2)-L(\theta_0)$ over the zoom $[-0.1,0.1]^2$ (a subset of the same 41-by-41 grid); no smoothing or fitting. Viewing angle, axis limits and colour scale are identical across panels (range 0 to 1.83 nats). The random draws are shared, but each model scales them by its own filter norms, so the four panels do not show one physical plane. One seed and one direction pair: an illustration, not a ranking. Zoom of the full-range figure: the same measured vertices restricted to $|a|,|b|\le0.1$ (9-by-9), with a common scale of its own.

Source: `tables/T_surface_vertices_seed0.csv`

## F_surfaces_centre_frozen_seed0_zoom_test

Loss slices of the four final seed-0 models under centre-frozen BatchNorm (statistics recalibrated once at the unperturbed weights on 2,000 training images, then held fixed): CIFAR-10 / ResNet-20-BN / SGD, 30 epochs, 1k-image test probe. Surfaces join the 9-by-9 measured vertices of $L(\theta_0+a d_1+b d_2)-L(\theta_0)$ over the zoom $[-0.1,0.1]^2$ (a subset of the same 41-by-41 grid); no smoothing or fitting. Viewing angle, axis limits and colour scale are identical across panels (range 0 to 1.67 nats). The random draws are shared, but each model scales them by its own filter norms, so the four panels do not show one physical plane. One seed and one direction pair: an illustration, not a ranking. Zoom of the full-range figure: the same measured vertices restricted to $|a|,|b|\le0.1$ (9-by-9), with a common scale of its own.

Source: `tables/T_surface_vertices_seed0.csv`

## F_surfaces_pointwise_seed0

Loss slices of the four final seed-0 models under pointwise-recalibrated batchnorm: CIFAR-10 / ResNet-20-BN / SGD, 30 epochs, 1k-image training probe. Surfaces join the 41-by-41 measured vertices of $L(\theta_0+a d_1+b d_2)-L(\theta_0)$ over $[-0.5,0.5]^2$; no smoothing or fitting. Viewing angle, axis limits and colour scale are identical across panels (range 0 to 4.18 nats). The random draws are shared, but each model scales them by its own filter norms, so the four panels do not show one physical plane. One seed and one direction pair: an illustration, not a ranking.

Source: `tables/T_surface_vertices_seed0.csv`

## F_surfaces_pointwise_seed0_test

Loss slices of the four final seed-0 models under pointwise-recalibrated batchnorm: CIFAR-10 / ResNet-20-BN / SGD, 30 epochs, 1k-image test probe. Surfaces join the 41-by-41 measured vertices of $L(\theta_0+a d_1+b d_2)-L(\theta_0)$ over $[-0.5,0.5]^2$; no smoothing or fitting. Viewing angle, axis limits and colour scale are identical across panels (range -0.00448 to 3.67 nats). The random draws are shared, but each model scales them by its own filter norms, so the four panels do not show one physical plane. One seed and one direction pair: an illustration, not a ranking.

Source: `tables/T_surface_vertices_seed0.csv`

## F_surfaces_pointwise_seed0_zoom

Loss slices of the four final seed-0 models under pointwise-recalibrated batchnorm: CIFAR-10 / ResNet-20-BN / SGD, 30 epochs, 1k-image training probe. Surfaces join the 9-by-9 measured vertices of $L(\theta_0+a d_1+b d_2)-L(\theta_0)$ over the zoom $[-0.1,0.1]^2$ (a subset of the same 41-by-41 grid); no smoothing or fitting. Viewing angle, axis limits and colour scale are identical across panels (range 0 to 0.463 nats). The random draws are shared, but each model scales them by its own filter norms, so the four panels do not show one physical plane. One seed and one direction pair: an illustration, not a ranking. Zoom of the full-range figure: the same measured vertices restricted to $|a|,|b|\le0.1$ (9-by-9), with a common scale of its own.

Source: `tables/T_surface_vertices_seed0.csv`

## F_surfaces_pointwise_seed0_zoom_test

Loss slices of the four final seed-0 models under pointwise-recalibrated batchnorm: CIFAR-10 / ResNet-20-BN / SGD, 30 epochs, 1k-image test probe. Surfaces join the 9-by-9 measured vertices of $L(\theta_0+a d_1+b d_2)-L(\theta_0)$ over the zoom $[-0.1,0.1]^2$ (a subset of the same 41-by-41 grid); no smoothing or fitting. Viewing angle, axis limits and colour scale are identical across panels (range -0.00448 to 0.304 nats). The random draws are shared, but each model scales them by its own filter norms, so the four panels do not show one physical plane. One seed and one direction pair: an illustration, not a ranking. Zoom of the full-range figure: the same measured vertices restricted to $|a|,|b|\le0.1$ (9-by-9), with a common scale of its own.

Source: `tables/T_surface_vertices_seed0.csv`

## F_surfaces_policy_comparison_rowwise_scale

Same seed-0 checkpoints, direction pair, filter scaling, 41-by-41 grid over $[-0.5,0.5]^2$ and 1k-image training probe; only the BatchNorm evaluation policy differs between rows (top: recalibrated at every vertex; bottom: recalibrated once at the centre, then frozen). Centre losses are identical under the two policies. Each row has its own common scale (pointwise 0 to 4.18; centre-frozen 0 to 171 nats); panels are comparable within a row, not across rows. Heights are CE above each model's own centre; no smoothing or fitting.

Source: `tables/T_surface_vertices_seed0.csv`

## F_surfaces_policy_comparison_rowwise_scale_test

Same seed-0 checkpoints, direction pair, filter scaling, 41-by-41 grid over $[-0.5,0.5]^2$ and 1k-image test probe; only the BatchNorm evaluation policy differs between rows (top: recalibrated at every vertex; bottom: recalibrated once at the centre, then frozen). Centre losses are identical under the two policies. Each row has its own common scale (pointwise -0.00448 to 3.67; centre-frozen 0 to 171 nats); panels are comparable within a row, not across rows. Heights are CE above each model's own centre; no smoothing or fitting.

Source: `tables/T_surface_vertices_seed0.csv`

## F_surfaces_policy_comparison_shared_scale

Same seed-0 checkpoints, direction pair, filter scaling, 41-by-41 grid over $[-0.5,0.5]^2$ and 1k-image training probe; only the BatchNorm evaluation policy differs between rows (top: recalibrated at every vertex; bottom: recalibrated once at the centre, then frozen). Centre losses are identical under the two policies. Both rows share one colour scale and z-range (0 to 171 nats). Heights are CE above each model's own centre; no smoothing or fitting.

Source: `tables/T_surface_vertices_seed0.csv`

## F_surfaces_policy_comparison_shared_scale_test

Same seed-0 checkpoints, direction pair, filter scaling, 41-by-41 grid over $[-0.5,0.5]^2$ and 1k-image test probe; only the BatchNorm evaluation policy differs between rows (top: recalibrated at every vertex; bottom: recalibrated once at the centre, then frozen). Centre losses are identical under the two policies. Both rows share one colour scale and z-range (-0.00448 to 171 nats). Heights are CE above each model's own centre; no smoothing or fitting.

Source: `tables/T_surface_vertices_seed0.csv`

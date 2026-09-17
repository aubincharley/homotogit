# Captions (standalone)

**fig_r_rg_vs_sdpoint_by_policy.** Seed-paired test-accuracy differences R − SDPoint (left) and RG − SDPoint (right) on CIFAR-10 / ResNet-20, for five training recipes and four BatchNorm evaluation policies at each method's native inference state. P0: checkpoint buffers (for SDPoint these mix training instances). P1 (primary): buffers reset and re-estimated as a cumulative average over the 50,000 clean training images, batch 500. P2: the same with batch 32. P3: author-style EMA pass (momentum 0.1, no reset) over a training loader with the recipe's batch size and augmentation. Markers show the mean ± SD of 3 seed-paired differences; grey dots are individual seeds. The historical 30-epoch recipes had prior test-set exposure.

**fig_accuracy_vs_training_cost.** P1 test accuracy (mean ± SD, 3 seeds) against measured training-loop time per run (T4 GPU-hours; excludes periodic evaluation, diagnostics and final calibration) for the three 160-epoch recipes. Published-schedule CBS also keeps 19 extra 3×3 depthwise convolutions at inference. After 160 epochs its σ = 0.9³¹, and these kernels are numerically identities.

**fig_learning_curves_p1_and_saved.** Top: P1 test accuracy after completed epochs 40, 80, 120 and 160, evaluated at each method's scheduled state at that epoch (SDPoint: full-resolution instance). Early points therefore describe a different network from the final one for R/G/RG/CBS. Bottom: the per-epoch saved-buffer evaluation on the current path (diagnostic). SDPoint's saved-buffer curve is not its calibrated performance.

**fig_gain_vs_plain_p1.** Mean seed-paired P1 test-accuracy gain over Plain (pp) for every arm and recipe, with sign counts over the 3 seeds.

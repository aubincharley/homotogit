# Experiment 1 — Gaussian warm starts and an intermediate continuation stage

Continue the existing research codebase. Implement and execute the experiment below, reusing Experiment 0 wherever scientifically valid. Keep the scope small: we now need to test adaptation to the original images, not launch another fixed-blur sweep or implement TV methods.

## Context and question

Experiment 0 trained 15 models: Gaussian input blur levels sigma in {0, 0.5, 1, 2, 3}, paired seeds {0, 1, 2}, and B = 14,040 optimizer updates per run. The official test set remained untouched. The existing report is `docs/results.md`, and the raw records are under `results/exp0_gaussian/`.

The unfiltered baseline reaches low training cross-entropy and approximately 84.11% target validation accuracy. Strongly blurred models perform substantially better on their own transformed distribution than on original images. Their original-image performance can decline during continued fixed-blur training.

Those observations do not establish whether blurred models are useful initializations for subsequent target training. Poor performance before adaptation is different from poor performance after adaptation.

Let x_i be the original image, y_i its label, T_sigma the existing Gaussian transformation with T_0 equal to the identity, and f_theta the classifier. Define

\[
L_\sigma(\theta)=\frac1n\sum_i
\ell_{\mathrm{CE}}\bigl(f_\theta(T_\sigma x_i),y_i\bigr).
\]

We now ask:

1. Does a short initial phase at sigma = 1 improve subsequent learning on L_0, at a fixed total update budget?
2. Does an intermediate phase at sigma = 0.5 add value compared with switching directly to sigma = 0?

These are empirical questions about optimization progress and generalization. Do not assume wider attraction basins, or claim that continuation requires the baseline to be trapped in a bad local minimum.

## 1. Inspect the existing configuration and checkpoints

Before training, inspect the actual code and run configurations. Record the architecture, normalization, split, preprocessing, optimizer, momentum, weight decay, batch size, and exact learning-rate schedule. In particular, explain whether the sharp loss decrease around 10,000–12,000 updates coincides with a learning-rate change.

Keep these settings unchanged for the main comparison. Retain the original normalization statistics, sample split, seeds, and Gaussian implementation. Apply each Gaussian level to the original image; never apply it to an already blurred image when transitioning to a finer level.

Inventory the saved checkpoints and distinguish model-only files from complete resumable training states. Eval metrics recorded every 500 steps do not imply that checkpoints exist at those steps.

For valid branching, preserve model parameters, optimizer state, scheduler state, global update counter, random-generator state, and the data-order/sampler position or an equivalent reproducible reconstruction. If mixed precision is used, preserve its scaler state too.

If an exact checkpoint at update 1,500 is unavailable, rerun only the required sigma = 1 prefix for each seed and save a complete state. This prefix must use the original B = 14,040 learning-rate horizon even though execution stops at 1,500; do not compress the schedule to the prefix length.

Reuse the sigma = 0 baselines if their configurations and paired initializations remain compatible. If a concrete incompatibility requires rerunning a baseline, explain it and rerun only the affected reference runs. Do not rerun the full 15-run sweep.

## 2. Prespecified main comparison

Use B = 14,040 updates and paired seeds {0, 1, 2} for all three arms:

| Arm | Updates 1–1,500 | Updates 1,501–3,000 | Updates 3,001–14,040 |
|---|---|---|---|
| A — direct baseline | sigma = 0 | sigma = 0 | sigma = 0 |
| W — direct warm start | sigma = 1 | sigma = 0 | sigma = 0 |
| P — intermediate continuation | sigma = 1 | sigma = 0.5 | sigma = 0 |

W and P must branch from the exact same complete sigma = 1 state after update 1,500 for each seed. Reuse that prefix computationally, but count its 1,500 updates in the budget of both methods.

The stage lengths are pragmatic pilot choices fixed before observing Experiment 1 results. They are not estimates of an optimal switch time from Experiment 0's validation peaks. Do not tune them separately by seed or silently change them after seeing an unfavorable result.

Use the existing learning rate indexed by global update count throughout. No learning-rate restart, no horizon rescaling, and no implicit optimizer reinitialization at a transformation change. Carry momentum at both transitions in this primary experiment. Record the policy explicitly; resetting momentum would be a different experiment.

Preserve the same minibatch sample-index sequence at equal global updates across arms for a paired seed. Branching must not accidentally change this sequence through different random-number consumption or evaluation calls.

Keep the original model normalization behavior. If the actual implementation contains running statistics, document how they evolve; do not add an unreported recalibration procedure.

Run seed 0 first to verify correct execution, then complete seeds 1 and 2 under the same specification. Proceed on scientifically negative results. Change the protocol only to fix a concrete implementation error, and clearly identify any invalid runs that must be repeated.

## 3. Minimal checks before the full runs

Verify the exact active sigma, learning rate, global counter, and optimizer state at each branch and transition. Confirm that sigma = 0 really supplies original images.

Use a small resume-versus-uninterrupted check if the existing code does not already establish correct resumption, especially when update 1,500 falls inside an epoch. Reuse existing checks when sufficient.

At a transition, evaluating original-image metrics immediately before and after changing the active training sigma, without updating weights or model state, must give the same result. Only evaluation on the active transformed distribution should change. This catches accidental coupling between target evaluation and the current training transform.

## 4. Measurements

Keep the existing fixed training probe subset and full validation split. Do not use the official test set.

Log minibatch training losses every 50 updates. At global update 0, every 500 updates, all stage boundaries, and the final update 14,040, record in eval mode:

- Target training CE and accuracy on original images from the fixed probe subset.
- Target validation CE and accuracy on original images.
- Training-probe and validation CE/accuracy at the currently active sigma, clearly labeled by sigma.
- Global update, active sigma, learning rate, stage identifier, and elapsed compute time.

For adaptation curves, add evaluations every 100 updates during the first 1,000 updates after each sigma transition. Include the boundary evaluation before any update on the new objective. These denser curves are descriptive; compare arms on common global checkpoints when making paired claims.

Save complete checkpoints at stage boundaries and at the end. Record each branch's parent checkpoint and configuration.

Report transformation cost and total method cost, including the shared prefix in the cost of each method. Separately report actual incremental compute consumed by this new experiment. Exclude machine suspension time from active-compute claims when identifiable, with the exception documented.

## 5. Primary analyses

### Equal total training budget

Plot target training CE and target validation accuracy against global updates for A, W, and P. Add vertical stage markers. Also provide target validation CE, since accuracy and confidence-related changes need not move together.

Report final metrics at B, individual seed results, mean and sample standard deviation, and the paired differences W minus A, P minus A, and P minus W. Express accuracy differences in percentage points. Three seeds support a preliminary comparison, not strong claims from tiny differences or error-bar overlap.

As a simple optimization diagnostic, report the first common checkpoint where target probe CE is at most 0.1 and at most 0.01. Mark a threshold as not reached if necessary. Use global updates including pretraining, and do not interpolate a precise crossing time from sparse measurements. These are shared target-loss thresholds, not estimated minima of different transformed tasks.

### Adaptation after transitions

Show target metrics immediately before switching and their trajectories afterward. For W, target training begins after update 1,500; for P, it begins after update 3,000.

Plots aligned by updates since the switch to original images can describe recovery, but must also state the different pretraining budgets. Do not treat equal numbers of target-stage updates as equal total computational cost.

Assess separately whether:

1. Performance on original images recovers quickly after switching.
2. W or P catches up with or exceeds A at equal global update count.
3. P improves on W enough to justify allocating 1,500 updates to the intermediate objective.

Recovery alone does not establish a net benefit. A benefit of P over A establishes usefulness of that procedure; P versus W addresses the additional contribution of the intermediate stage.

## 6. Interpretation and scope

- Do not infer that a poor pre-switch target score proves destruction of transferable features.
- Do not infer that final near-perfect training accuracy rules out optimization speedups at smaller budgets.
- Do not infer that 1,500 updates is an optimal warm-start length or that sigma > 1 cannot help.
- This experiment tests early warm starts. It does not establish whether late coarse checkpoints are worse after adaptation; that would require a separate controlled comparison.
- A negative result applies to the tested transformation, schedules, optimizer, and data. A positive result does not prove landscape smoothing or convergence to a global optimum.
- Do not expand this phase into a sweep over sigma, stage lengths, learning rates, momentum resets, TV, wavelets, new datasets, or architectures.

## 7. Deliverables and execution

Implement the small branching/continuation extension, perform the necessary checks, and execute the six new W/P trajectories plus any required prefix reconstruction. Keep Experiment 0 raw results intact.

Save run configurations, lineage, metrics, and aggregation under a distinct experiment directory, such as `results/exp1_gaussian_warmstart/`. Produce a concise report such as `docs/exp1_warmstart.md`, with plots and commands needed to reproduce the experiment. Add an interpretation note identifying which categorical claims in the earlier report were unsupported; preserve its measured numbers.

In the final report, answer the three questions in Section 5 directly, distinguish measurements from hypotheses, and state the limits of the three-seed study. If compute or files are genuinely unavailable, complete the runnable setup and identify exactly what remains unexecuted. Never fabricate results or stop at a plan when execution is available.

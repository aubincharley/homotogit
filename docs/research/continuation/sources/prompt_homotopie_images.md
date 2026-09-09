# Research codebase: continuation through image simplification

Build a small, reproducible research codebase to study continuation methods for neural-network training. The primary research direction is **input-image transformations that progressively relax a simplification or complexity constraint**. Gaussian spatial smoothing is the first transformation to implement. Later we want to compare it with transformations based on total variation and, potentially, operational compression budgets.

The priority is understanding optimization and transfer to the original classification problem, rather than achieving state-of-the-art CIFAR accuracy.

**Immediate scope:** implement Gaussian input smoothing and Experiment 0, which trains independent models at fixed smoothing levels. Prepare extension points for other transformation families and continuation schedules. Do not implement TV solvers, compression methods, feature-map smoothing, or adaptive continuation in this first phase. Do not run a full continuation experiment yet.

## 1. Common mathematical setup

Let

\[
\mathcal D=\{(x_i,y_i)\}_{i=1}^{n},
\qquad x_i\in[0,1]^{3\times H\times W},
\]

and let \(f_\theta\) produce logits in \(\mathbb R^{N_{\mathrm{classes}}}\). Write the target empirical loss as

\[
L_{\mathrm{target}}(\theta)
=\frac1n\sum_{i=1}^{n}
\ell_{\mathrm{CE}}(f_\theta(x_i),y_i).
\]

For an image-transformation family \(T_\eta\), define

\[
L_\eta(\theta)
=\frac1n\sum_{i=1}^{n}
\ell_{\mathrm{CE}}(f_\theta(T_\eta x_i),y_i).
\]

The native parameter \(\eta\) depends on the family. For Gaussian filtering it is \(\sigma\geq0\), with the target at \(\sigma=0\). For a relative complexity budget it may be \(t\in[0,1]\), with the target at \(t=1\). The framework must not assume that every native parameter decreases toward zero.

Keep three objects separate:

1. The transformation family and its mathematical parameter.
2. The continuation schedule/controller, which chooses that parameter during training.
3. The optimizer and its learning-rate schedule.

Changing the first object changes the family of objectives. Changing the second changes how the same family is traversed.

All initial transformations must be label-blind and deterministic given their configuration and image. Apply each transformation to the **original image**, not to the previously transformed image: increasingly detailed stages must be able to recover information from the original input.

## 2. Gaussian baseline

The first family is

\[
T_\sigma x=G_\sigma*x,
\qquad T_0x=x,
\]

where the normalized isotropic Gaussian is applied spatially and independently to each color channel. Report \(\sigma\) in pixel units.

For the continuous Fourier transform expressed in angular frequency \(\omega\),

\[
\widehat{T_\sigma x}(\omega)
=e^{-\sigma^2\|\omega\|^2/2}\widehat x(\omega).
\]

The associated heat-time parameter is

\[
a=\sigma^2/2,
\qquad
\widehat{T_a x}(\omega)=e^{-a\|\omega\|^2}\widehat x(\omega).
\]

The continuous heat operators satisfy \(T_{a_1}T_{a_2}=T_{a_1+a_2}\). This motivates alternative schedule parameterizations. **Do not claim that this exact semigroup identity holds for a finite, truncated, padded discrete Gaussian implementation.**

For the initial implementation:

- Use a deterministic, normalized, separable Gaussian filter with documented reflection padding and spatial support.
- Prefer one fixed support large enough for the maximum configured \(\sigma\), so changing a kernel-size rounding rule does not introduce extra discontinuities along the parameter path.
- Implement \(\sigma=0\) explicitly as the identity, avoiding division by zero or a residual blur.
- Work in floating-point image space, before channel normalization. Do not quantize filtered training images back to 8-bit values.
- Do not filter across channels or class logits.
- Verify identity, preservation of constant images, dimensions, deterministic behavior, and plausible attenuation of spatial oscillations. Document boundary effects.
- Do not implement hard Fourier cutoffs or projections.

The proposed continuation will eventually decrease \(\sigma\), reusing learned weights between levels. A fixed-\(\sigma\) run is the first special case to support.

## 3. Future transformations: preserve fidelity under a complexity budget

This section specifies the scientific direction and extension requirements; it does **not** request implementation of these additional transformations now.

Given a scalar image-complexity functional \(C\), the intended construction is

\[
T_t^C(x)
\in\operatorname*{arg\,min}_{z\in[0,1]^{3\times H\times W}}
\frac12\|z-x\|_2^2
\quad\text{subject to}\quad C(z)\leq tC(x),
\qquad t\in[0,1].
\]

This chooses the closest admissible reconstruction. Maximizing \(C(z)\) subject only to \(C(z)\leq tC(x)\) would not tether the reconstruction to the original image; it would not even ensure that the original image is recovered at \(t=1\).

For a new complexity functional, explicitly specify its domain, minimum, feasible budgets, reconstruction fidelity, and any tie-breaking rule. Nonconvex complexity functionals need not yield a unique or continuous path. Do not assume that all families possess Gaussian semigroup or nesting properties.

### 3.1 First planned extension: a TV budget

A concrete choice is channelwise discrete isotropic total variation,

\[
\operatorname{TV}(z)
=\sum_{c,p}
\sqrt{(D_1z_c)_p^2+(D_2z_c)_p^2},
\]

with a documented finite-difference convention, pixel spacing, and zero-normal-difference boundary treatment on a connected image grid. This TV is a roughness functional on pixel values, not a total-variation distance between probability distributions.

Then

\[
T_t^{\mathrm{TV}}(x)
=\operatorname*{arg\,min}_{z\in[0,1]^{3\times H\times W}}
\frac12\|z-x\|_2^2
\quad\text{subject to}\quad
\operatorname{TV}(z)\leq t\operatorname{TV}(x).
\]

The feasible set is closed and convex, and the quadratic fidelity gives a unique minimizer. The exact finite-dimensional solution varies continuously with the budget. At \(t=1\), it is exactly \(x\). At \(t=0\), it is a spatially constant image in each channel, equal to that channel's mean. If \(\operatorname{TV}(x)=0\), return \(x\) at all levels.

The zero-budget endpoint is a mathematical reference; it is not a mandatory starting point for classification training. A positive initial budget will generally need to be selected experimentally.

A related implementation route is the penalized TV problem

\[
\operatorname*{arg\,min}_{z\in[0,1]^{3\times H\times W}}
\frac12\|z-x\|_2^2+\lambda\operatorname{TV}(z).
\]

A common \(\lambda\) and a common relative TV budget \(t\) are different experimental choices. Enforcing a particular relative budget can require image-dependent calibration of \(\lambda\). A future solver must report achieved TV, fidelity, feasibility error, numerical tolerance, and transformation cost; it must not silently claim an exact budget after an unconverged approximate solve.

### 3.2 Later possibility: an operational rate budget

Shannon rate-distortion is a function of a source distribution, distortion criterion, and reconstruction channel:

\[
R(D)=\inf_{P_{\widehat X\mid X}:\,\mathbb E[d(X,\widehat X)]\leq D}
I(X;\widehat X).
\]

It is not automatically a scalar complexity \(C(x)\) of a single image. An operational implementation must specify a codec or representation, an actual code-length/rate measure, a decoder, and a distortion criterion. One possible future definition is to seek the least-distorted reconstruction achievable within a bit budget.

Record actual bits per image or bits per pixel. A codec quality setting is not itself a measured rate. A codec-based family can have discrete jumps, a nonzero minimum rate, and a lossy maximum-quality setting; explicitly handle whether and how its target endpoint recovers the original image. Do not claim a continuous homotopy or a Shannon-optimal rate-distortion solution merely because a compression parameter is varied.

Do not equate visual smoothing with a strict decrease in Shannon information. An ideal Gaussian Fourier multiplier is nonzero at every finite frequency: without quantization, observation noise, or another irreversible operation, the transformation can remain injective. Roughness, numerical conditioning, coding rate, and mutual information are different notions.

### 3.3 What would make a simplification useful for learning?

Low TV or a low coding rate does not imply an easier classification problem. A transformation can remove the details needed to distinguish classes. Pixel fidelity also does not guarantee preservation of label-relevant information.

Treat usefulness as a hypothesis to test through optimization behavior, discriminative performance, and eventual transfer to the original objective. Distinguish a well-posed convex **image transformation problem** from the generally nonconvex **neural-network training problem**.

The framework should support reporting actual retained complexity, such as \(\operatorname{TV}(T_\eta x)/\operatorname{TV}(x)\), and reconstruction MSE on fixed image subsets. Handle zero-TV images explicitly. Such measurements can help calibrate different transformations, but equal TV or equal MSE does not mean equal semantic information or equal learning difficulty. Gaussian \(\sigma\) is not an exact per-image TV budget.

## 4. Dataset and classifier

Start with CIFAR-10, downloaded from a standard trusted source. Use a fixed, reproducible, preferably stratified split of the official training set into approximately 45,000 training and 5,000 validation examples. Keep the official 10,000-image test set untouched until final comparisons.

Make switching to CIFAR-100 possible without redesigning the pipeline. Avoid random crops, rotations, color jitter, and other augmentations in the initial study. Compute any fitted preprocessing statistics on the training subset only. Apply Gaussian filtering before ordinary channel normalization, using the same normalization convention in all runs.

Use a small CIFAR-style ResNet-20 with GroupNorm, or an equivalently lightweight residual classifier. Document the exact architecture and GroupNorm configuration. The classifier must have 10 outputs for CIFAR-10 and 100 for CIFAR-100.

GroupNorm avoids BatchNorm running-statistic state when the input distribution changes. It does not remove the effects of the input transformation itself.

Keep the initial network unmodified internally. Future feature-map filtering and activation homotopies should remain possible as separate controlled experiments. In particular, the feature-smoothing method in *Curriculum by Smoothing* is distinct from the present input-only baseline. Do not silently replace or combine the two interventions.

## 5. Optimization and reproducibility

Use minibatch SGD with momentum as the primary optimizer, initially with batch size 128 and momentum 0.9. Choose and document a conventional CIFAR learning rate and schedule. Any weight decay must be fixed across conditions and reported separately from the unregularized cross-entropy objective.

Hold architecture, batch size, optimizer settings, learning-rate schedule, and training-update budget fixed across the main comparisons. Use paired initialization seeds and the same sequence of minibatch sample indices whenever possible. A shared seed alone is insufficient if different transformations consume random numbers differently.

For future continuation, index the learning-rate schedule by global optimization step, independently of transformation stages. Do not implicitly restart it at every stage. Make the treatment of SGD momentum at stage boundaries explicit: carrying it and resetting it are different procedures. No such stage transition is required in Experiment 0.

Report both gradient-update counts and elapsed time, including transformation/preprocessing cost separately when possible. Equal update counts are an optimization-budget control; they are not automatically equal total computational cost, particularly for future iterative TV transformations.

Use three paired seeds for the first complete study, with configuration support for five or more. Log exact configurations, software versions, split indices, seeds, metric definitions, and checkpoints. Do not introduce another varying regularization mechanism.

## 6. Experiment 0: independent training at fixed Gaussian levels

Train separate models from scratch at fixed \(\sigma\) values. The initial candidate grid is

\[
\sigma\in\{0,0.5,1,2,3\}\quad\text{pixels}.
\]

First show a fixed, class-diverse set of training images at these levels. Record the grid used and any adjustments made during the visual/pilot phase, before the main comparison. Do not select a different favorable range separately for each reported method or use the test set to choose it.

For every level, record at shared checkpoints:

1. Training cross-entropy and accuracy for that run's transformed training problem.
2. Validation cross-entropy and accuracy using the same transformation level.
3. **Target training cross-entropy** on the original, unfiltered images at the same weights, using a fixed documented training probe subset if full evaluation is expensive.
4. Target validation cross-entropy and accuracy on original images.
5. Final metrics at a common update budget and paired-seed variability.
6. Transformation statistics on the fixed image subset: reconstruction MSE and retained TV ratio, with the TV convention specified.

Distinguish minibatch losses logged during training from evaluation-mode metrics computed on fixed subsets. Name and label both clearly.

### Interpretation limits

Experiment 0 asks how fixed transformations affect training progress and task performance. It does not by itself establish that the transformed losses have wider attraction basins or that continuation will improve the final result.

- Validation performance measures generalization on a particular transformed task; it is not a direct measurement of optimization difficulty.
- Raw losses from different transformed tasks may have different attainable minima. A quick plateau is not evidence of a better optimization problem, and information loss can raise the best attainable classification error.
- Original-image evaluation of a model trained only on transformed images measures transfer under a distribution change. It is useful, but it is not a substitute for actually testing a later warm start on the target loss.
- Small gradient norms alone do not imply useful convergence.
- Common optimizer settings provide a controlled primary comparison, not proof that each task has been optimized as effectively as possible. If instability drives a result, document it and use a clearly separated, predeclared learning-rate sensitivity check rather than silently tuning individual conditions.

Produce curves of transformed training cross-entropy versus updates, target training cross-entropy versus updates, and both validation evaluations. Provide numerical summaries with uncertainty across seeds. Do not merge these quantities into one unsupported score of "ease."

## 7. Extension architecture

Keep the implementation small. A configuration-driven interface is enough; avoid building an elaborate generic framework before it is needed.

Provide separable components for:

- **Transformation family:** apply a named transformation at a native parameter value to original images; expose its target endpoint and configuration.
- **Complexity/distortion measurements:** compute diagnostics independently of the transformation. Do not require every family to have an exact scalar complexity certificate.
- **Schedule/controller:** return the next native parameter or stage from global training progress and, in the future, diagnostic state.
- **Trainer/evaluator:** train or evaluate the classifier at any fixed transformation parameter without rebuilding the whole experiment.

Future per-image budgets may require sample identifiers, image-dependent calibration, and metadata returned by the transformation. Cache transformed data only under an unambiguous key containing image identity, family configuration, parameter, numerical conventions, and solver tolerance where applicable. Budget or codec settings must not be conflated with achieved complexity or rate.

Future network/activation or parameter-space homotopies will require separate model/objective extension points. A data-transform interface alone does not implement those families. Learned continuation controllers and learned representations of the solution path are further changes to the training procedure, not additional image filters.

## 8. Future continuation schedules and diagnostics

Prepare, but do not yet run, stagewise and smoothly parameterized schedules:

- Piecewise-constant native parameters with specified update counts per stage.
- A Gaussian standard deviation linear in training progress.
- A geometric Gaussian schedule, with an explicit final identity stage: a finite geometric sequence does not reach zero by itself.
- A schedule linear or geometric in heat time \(a=\sigma^2/2\).
- Increasing relative complexity budgets for future TV transformations.
- Later adaptive schedules based on optimization state.

Make initial level, stage locations, updates per stage, and total update budget independently configurable. Allocate a documented terminal training period to the exact target objective when full continuation is eventually studied.

For later transition diagnostics, evaluate gradients at the **same weights**, on the **same fixed probe examples**, with controlled model state:

\[
g_k(\theta)=\nabla_\theta L_{\eta_k}(\theta),
\qquad
g_{k+1}(\theta)=\nabla_\theta L_{\eta_{k+1}}(\theta).
\]

Useful candidates include the next-objective gradient norm, gradient differences, and their cosine similarity. Report the gradient norms alongside any cosine. Near a stationary point, the old gradient may be close to zero: the cosine is then undefined or unreliable. Use an explicit norm threshold and mark it unavailable; do not turn it into a misleading number through a silent denominator adjustment.

These diagnostics are not proofs of common attraction basins. Later experiments should compare direct target training, the complete continuation path, and a coarse-to-target warm start that skips intermediate stages, with matched update budgets and initializations. That comparison isolates whether intermediate objectives add value beyond the coarse initializer.

## 9. Immediate deliverables

1. Obtain and split CIFAR-10 reproducibly.
2. Implement and visually verify the Gaussian input transformation, including its exact identity endpoint.
3. Instantiate the lightweight GroupNorm residual classifier.
4. Support fixed-parameter runs through interfaces that can later accommodate other families and schedules.
5. Execute Experiment 0 where the available compute allows it, with clearly documented runtime settings; if execution is blocked, provide runnable commands and state exactly which runs remain unexecuted. Never fabricate curves or numerical results.
6. Generate clear plots, machine-readable metrics, and a short interpretation separating optimization progress, transformed-task generalization, and transfer to the target task.
7. Document the planned TV-budget and operational-rate extensions without implementing them in this phase.

The first study should determine whether input Gaussian smoothing is a promising member of the proposed family of simplifications. A negative result is scientifically informative. It does not invalidate every continuation method, and a positive result does not establish a general convergence theorem.

## 10. Relevant sources and distinctions

- Sinha, Garg, and Larochelle, [*Curriculum by Smoothing*](https://papers.nips.cc/paper_files/paper/2020/file/f6a673f09493afcd8b129a0bcf1cd5bc-Paper.pdf), NeurIPS 2020, especially Section 3 and Table 7. This studies internal feature smoothing and includes input-smoothing ablations; it does not justify assuming input blur will help.
- Rudin, Osher, and Fatemi, [*Nonlinear total variation based noise removal algorithms*](https://www.eng.utah.edu/~cs7640/readings/PhysicaRudinOsher.pdf), 1992. A foundational reference for TV-based fidelity/regularity tradeoffs in image restoration; the budget-indexed classification curriculum above is the experimental proposal, not a training guarantee from that paper.
- Bernd Girod, Stanford EE398A, [*Rate Distortion Theory*](https://web.stanford.edu/class/ee398a/handouts/lectures/04-RateDistortionTheory.pdf). Use the distinction between a source rate-distortion function and a practical codec's rate/distortion measurements.

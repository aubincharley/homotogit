# Notes for sections not yet completed

Working document. These plans were removed from the paper PDF on 14 September
2026, because the corresponding work has not been done. The paper keeps one red
sentence at each location (§4.2 transfer, §5 theory, §6 visualization, §7
discussion, abstract).

## Transfer (Aubin, Alexandre) — §4.2

Result table to fill only from completed, verified runs: final test accuracy (%),
mean ± sample SD over seeds, at the native filter-free endpoint; columns Plain,
Resolution, Gaussian, Combined; one row per dataset / architecture / optimizer.
Keep *not yet run*, *failed k/n* and *not applicable* distinct. Do not reuse the
test-selected CIFAR-10 exploration as independent confirmation.

Candidate rows (placeholders, not commitments): CIFAR-10 ResNet-20 SGD; STL-10
ResNet-20 SGD; CIFAR-10 ResNet-20 AdamW; CIFAR-10 additional CNN (e.g. VGG-11).

Decisions each transfer run must record:

- Optimizer hyperparameters chosen on training/validation data, with comparable
  tuning effort across methods; same update budget and pinned assets within a
  comparison; runtime measured, not inferred.
- Insertion mapping. Implemented in `continuation_core`:

  | Architecture | Conv-output sites | Post-ReLU sites | Reduction point | Status |
  |---|---|---|---|---|
  | ResNet-20 (BN) | 19 (stem + both convs of each block) | 10 (stem ReLU output + each block output) | input of block 2 | reference, matches executed runs |
  | VGG-11 (BN) | 8 conv outputs | 8 (every ReLU output) | none defined | implemented, never run; resolution methods raise an error until a point is named |

  For VGG-11 the ReLU rule differs from ResNet's block outputs; report the choice.
- STL-10 images are 96×96: the map entering block 2 of ResNet-20 has side 96, so
  reference resolution, reduced sides and σ in pixels must be chosen explicitly.
- AdamW: learning rate, weight decay and schedule are explicit inputs.

## Theory (Idriss) — §5

A tractable first setting is a regression model whose components become available
at different times, looking for conditions under which delayed fitting increases
empirical error while decreasing expected prediction error (early-stopping analyses,
Ali et al. 2019, are a starting point). Such a result would illustrate a mechanism;
it would not prove the explanation for max-pooling, BatchNorm or cross-entropy. A
smaller upper bound on risk does not order two actual risks.

## Visualization (Max) — §6

Three questions: loss at fixed weights under different intervention states; change
of the trajectory at schedule transitions; sensitivity of the final weights.

- Common plane `F_s(a,b) = L_s(θ_ref + a d1 + b d2)` with training subset, reference
  point and directions fixed across states; joint PCA directions reported with
  explained variance and per-checkpoint projection residuals. A projected
  checkpoint is off the plane: the background loss is not its actual loss.
- Filter-normalized perturbations around final checkpoints (Li et al. 2018).
- BatchNorm policy always stated (running statistics vs per-batch statistics on
  cloned buffers); no buffer modified. A flat 2-D slice is not a generalization
  argument (Dinh et al. 2017).
- The exploratory checkpoints are not versioned; new checkpoints around the
  transitions are needed. Tools: `continuation_core` (`evaluate`,
  `export-trajectory`, `pca-plane`, `plane-loss`, `perturb`).

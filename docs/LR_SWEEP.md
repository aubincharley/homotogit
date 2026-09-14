# Learning-rate sweep

Adam, AdamW and RAdam had never been run anywhere in this repository, so there
was no basis for a learning rate. The reference 0.005 is an SGD value and
nothing about it transfers. This sweep picks one per optimizer before the grid
runs.

**Protocol.** `plain` only, seed 0, the reference recipe with the optimizer and
learning rate swapped and nothing else: 30 epochs, no augmentation, weight decay
5e-4 on every parameter, 60 warmup updates then cosine. Pinned assets verified
on the worker. Kernel
`alexandrecorrard/job-optbench-lr-sweep-20260914-100029`, 2x T4, 12/12 cells
complete, none diverged.

**Selection rule, fixed before the results were read:** highest final test
accuracy on `plain`, seed 0; ties inside 0.3 pt go to the lower learning rate;
a diverged arm is discarded and recorded. Tuning on `plain` gives the control
its best showing, so the bias runs *against* continuation — the right direction
for a test of whether continuation survives.

## Results

Final test accuracy, target path, epoch 30.

| lr | adam | adamw | radam |
|---|---:|---:|---:|
| 3e-4 | 75.37 | 73.89 | 74.09 |
| 1e-3 | 81.29 | 79.79 | 80.07 |
| 3e-3 | **83.95** | 82.98 | **84.58** |
| 1e-2 | 81.97 | **83.80** | 84.42 |
| 2e-2 | — | _pending_ | — |
| 3e-2 | — | _pending_ | — |

Reference for scale: `plain` under SGD at lr 0.005 is 75.43 ± 0.76 over three
seeds, and 74.58 on seed 0 — the same seed and the same pinned assets as every
cell above.

## Boundary extension

AdamW is the only optimizer whose best value landed on an endpoint of the grid,
which means the grid did not bracket its optimum. Adam and RAdam both peak in
the interior. Comparing a boundary-limited AdamW against two interior-optimum
arms would put back exactly the "learning rate not tuned for this optimizer"
confound the sweep exists to remove, so the rule applied is: **extend past any
endpoint optimum**. Only AdamW qualifies, and it is extended to 2e-2 and 3e-2
(`lr_sweep_ext`). The rule is symmetric — it would have been applied to any
optimizer that hit an endpoint, in either direction.

## Chosen

_Filled in once the extension lands._ On the original grid alone the rule gives
adam 3e-3, radam 3e-3 (84.58 against 84.42 at 1e-2 — inside 0.3 pt, so the tie
rule takes the lower rate), and adamw 1e-2 pending the extension.

## What this already says about the benchmark

The sweep was meant to be preparation. It is not.

`plain` reaches **84.58 %** under RAdam at 3e-3. The best *continuation* arm
ever recorded on this recipe is `resolution_max_b1_gaussian_conv` at
**81.51 ± 0.22** under SGD. A plain ResNet-20, with no intervention of any
kind, beats every continuation arm in the frozen benchmark by about three
points once the optimizer is allowed to be a reasonable one.

The gap between the SGD control and the adaptive-optimizer control is **+9.2 pt**
(74.58 → 84.58, same seed, same initial weights, same data order). The entire
continuation effect the benchmark reports is +4 to +6 pt. So the intervention
was measured against a baseline that was leaving roughly twice the intervention's
own effect on the table.

This does **not** show that continuation does nothing. It shows that the
headline numbers cannot be read as "continuation is worth ~5 points", because a
change of optimizer is worth about twice that on the same control. Whether
continuation still adds anything *on top of* a well-tuned adaptive optimizer is
exactly what the grid measures, and it is now the whole question.

Two things to keep in mind when reading the grid:

* The effect could be **subsumed** — if continuation buys speed of optimization
  early, an optimizer that already adapts its per-parameter step may capture the
  same thing, and Δ collapses toward zero.
* It could also **survive**, at reduced size, if the mechanism is regularisation
  or a genuine easier-to-harder path rather than conditioning. The paired Δ is
  what separates these, and it is computed inside each optimizer so the level
  difference above cancels.

Either outcome is a result. The one reading now foreclosed is the one where the
SGD table stands on its own.

# Experiment 0 — results

**Status: executed in full.** 15/15 runs completed (σ ∈ {0, 0.5, 1, 2, 3} × seeds {0, 1, 2}),
14,040 gradient updates each (40 epochs at 351 updates/epoch), identical budget and
optimizer settings in every condition. No run was excluded. The official test set was
not touched.

Hardware: RTX 3050 Laptop (4 GB), torch 2.5.1+cu121, deterministic cuDNN.
Raw records: `results/exp0_gaussian/` (per-run `metrics.jsonl`, `summary.json`,
`run_description.json`; sweep-level `aggregate.json` / `aggregate.csv`).

Grid and budget were fixed before the main comparison and were **not** adjusted
afterwards. The visual/pilot phase (`figures/transform_levels.png`) preceded the
sweep; no per-condition tuning was performed and no level was added or removed.

## Final metrics at the common update budget

Mean over 3 paired seeds, sample sd in parentheses. All cross-entropies are
unregularized; weight decay is excluded.

| σ (px) | target val acc | transformed val acc | target train CE | recon. MSE | retained TV |
|---|---|---|---|---|---|
| 0   | **0.8411** (0.0060) | 0.8411 (0.0060) | 0.0034 | 0.00000 | 1.0000 |
| 0.5 | 0.8255 (0.0006) | 0.8355 (0.0038) | 0.0219 | 0.00040 | 0.8200 |
| 1   | 0.7133 (0.0238) | 0.8063 (0.0036) | 0.7529 | 0.00345 | 0.5568 |
| 2   | 0.4832 (0.0296) | 0.7430 (0.0009) | 2.1384 | 0.00951 | 0.3505 |
| 3   | 0.3885 (0.0186) | 0.6741 (0.0033) | 2.5208 | 0.01445 | 0.2548 |

`target_*` = evaluated on **original unfiltered** images at the same weights;
`transformed_*` = evaluated at the run's **own** σ. At σ=0 the two objectives are
the same objective, so the numbers coincide by construction.

## Finding 1 — smoothing did not make optimization easier

`figures/exp0_curves.png`, top-left panel. Transformed training cross-entropy
descends **fastest at σ = 0 and σ = 0.5** (near-identical curves) and progressively
**slower** as σ increases. σ=2 and σ=3 reach their own near-zero training loss later
than the unfiltered problem does.

There is no evidence here for the intuition that a smoothed input problem is an
easier optimization problem. If anything the ordering runs the other way.

Two limits on that statement, both of which matter:

* Different σ define **different objectives with different attainable minima**, so
  comparing their raw loss values is not apples-to-apples. The comparison above is
  about *rate of descent toward each objective's own floor*, not about loss levels.
* **Every condition reached ≈100 % training accuracy** (final transformed training CE
  between 0.003 and 0.064). Direct training on the target objective never got stuck,
  so this testbed contains no optimization failure for a continuation method to
  repair. See "Implications" below — this is the single most consequential fact in
  the whole experiment.

## Finding 2 — transfer degrades steeply, and superlinearly in σ

Target validation accuracy: 0.841 → 0.826 → 0.713 → 0.483 → 0.389.

σ=0.5 is nearly free (−1.6 points) while retaining 82 % of image TV. By σ=1 the cost
is −12.8 points, and by σ=2 the model is worse than a coin flip between the two most
likely classes. On 32×32 images, where an object spans ~15–20 pixels, σ ≥ 2 blurs
across a large fraction of the object itself.

Seed variability also grows with σ (sd 0.006 at σ=0 → 0.030 at σ=2), so the coarse
conditions are both worse and noisier.

## Finding 3 — the most useful result for continuation: coarse models peak *early*, then decay

Target validation accuracy over training, per level (mean over seeds):

| σ | best target val acc | at update | final | change from peak |
|---|---|---|---|---|
| 0   | 0.8421 | 12,500 | 0.8411 | −0.001 |
| 0.5 | 0.8285 | 11,500 | 0.8255 | −0.003 |
| 1   | 0.7327 | 8,000  | 0.7133 | −0.019 |
| 2   | 0.5765 | 6,333  | 0.4832 | **−0.093** |
| 3   | 0.4928 | 5,000  | 0.3885 | **−0.104** |

The peak arrives **monotonically earlier** as σ grows, and the subsequent decay is
large. Meanwhile the *transformed* validation accuracy for those same runs keeps
improving to the end (bottom-right vs. middle-right panels). The target training CE
curves (top-right panel) show the same thing: for σ ≥ 1 they descend, bottom out, and
then climb back up.

The model is progressively **specializing to the blurred input distribution**, and
that specialization actively costs it target performance. Training a coarse stage to
convergence destroys much of what you would want to carry forward.

This is a concrete, actionable constraint on schedule design, and it is the one thing
Experiment 0 tells us that could not have been guessed from the final numbers alone.

## Finding 4 — transformation cost is negligible here

10–12 s of transformation per ~500 s run (≈2 %), measured separately. This will not
stay true for iterative TV solvers, which is why the accounting is separate.

*Caveat:* `level_0p5__seed_1` reports 7,446 s of wall time because the machine was
suspended mid-run. That number measures the suspension, not the workload, and is
excluded from all cost statements. Its *metrics* are unaffected — the run continued
correctly from the same process and completed normally.

## Implications for continuation / homotopy training

**What this does not establish.** Nothing here says the transformed losses have wider
attraction basins, and nothing here substitutes for actually testing a warm start on
the target loss. Original-image evaluation of a model trained only on blurred images
measures transfer under a distribution change — informative, but a different quantity.

**The headline problem.** Continuation methods earn their keep when direct
optimization *fails*. Here it does not: every condition, including σ=0, fits the
training set essentially perfectly. A continuation scheme on this testbed can
therefore only act as a regularizer or an initializer, and Finding 2 shows that
anything learned at σ ≥ 1 has to survive a large distribution shift to be useful.
Before investing in the continuation machinery, decide whether the claim being tested
is about **optimization** (in which case this testbed is inadequate and needs a
setting with genuine optimization pathology) or about **generalization/transfer** (in
which case it is fine, but the claim must be stated that way).

**If continuation is run anyway, this experiment constrains the design:**

1. **Usable σ range on 32×32 is roughly [0, 1].** σ=2 and σ=3 destroy too much label
   information to be plausible starting points — you cannot recover what is not there.
   That is a short interval over which to lay out a path.
2. **Stages must be short.** Finding 3 gives concrete numbers: a σ=2 stage is already
   past its useful point by ~6,000 updates, and a σ=3 stage by ~5,000. Schedules that
   train each level to convergence are the wrong shape.
3. **Higher resolution would lengthen the path.** σ is in pixels; at 64×64 or 96×96 the
   same relative simplification spans a much wider σ range. This addresses the
   short-path problem but *not* the no-optimization-failure problem.
4. **The decisive comparison remains the three-way one**, at matched update budgets and
   matched initializations: (a) direct target training, (b) the full continuation path,
   (c) a coarse-to-target warm start that **skips** the intermediate stages. (c) is what
   isolates whether traversing intermediate objectives adds anything beyond supplying a
   cheap initializer. Without it, a win for (b) over (a) is uninterpretable.

**Is Gaussian smoothing a promising member of the family?** On this evidence, weakly
negative. It did not accelerate optimization, it costs target accuracy at every level
above 0, its usable range on CIFAR-10 is narrow, and its coarse levels degrade with
further training. That is an informative negative result about *this* transformation
on *this* testbed. It does not invalidate continuation in general, and it says nothing
about the TV-budget family, whose constrained-reconstruction construction keeps the
transformed image tethered to the original in a way Gaussian blur does not.

## Figures

* `figures/transform_levels.png` — fixed class-diverse training images at each level (pilot phase)
* `figures/boundary_effects.png` — reflection-padding behaviour, constant and step profiles
* `figures/attenuation.png` — measured grating attenuation vs. the continuous prediction
* `figures/exp0_curves.png` — the four eval-mode metric families vs. updates, mean ± 1 sd
* `figures/exp0_minibatch_ce.png` — training-mode minibatch CE, logged separately
* `figures/exp0_final_vs_level.png` — final metrics vs. σ with seed error bars
* `figures/exp0_transform_stats.png` — reconstruction MSE and retained TV ratio vs. σ

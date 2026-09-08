# exp4 -- the activation homotopy on the whole dataset

**The headline result of the activation axis, and it is negative on every
metric.** CIFAR-10 in full (50 000 images), 80 epochs, three seeds, one GPU
budget per arm.

Reproduce:

    make push-act ACT=full80_baseline SEEDS=0,1,2
    make push-act ACT=full80_steps    SEEDS=0,1,2

Configs: `src/cifarbase/configs/full80_baseline.yaml`, `full80_steps.yaml`.
Raw runs (untracked, gitignored): `out/exp4_full80_3seeds/`.

## Result

| metric | baseline | staircase | difference | |
|---|---|---|---|---|
| test acc | 0.9483 ± 0.0004 | 0.9364 ± 0.0005 | **-0.0119** | 19.2 σ |
| test loss | 0.2053 ± 0.0009 | 0.2169 ± 0.0009 | **+0.0116** | 9.2 σ |
| gen gap | 0.0516 ± 0.0004 | 0.0592 ± 0.0005 | **+0.0075** | 12.2 σ |
| train acc | 0.9999 ± 0.0000 | 0.9956 ± 0.0002 | -0.0044 | 27.4 σ |

Per seed, with no overlap on either metric:

    baseline    acc 0.9485  0.9475  0.9489     loss 0.2070  0.2049  0.2041
    staircase   acc 0.9358  0.9361  0.9373     loss 0.2153  0.2171  0.2185

Gradient updates to reach a level, staircase relative to baseline:

    85%  2.28x      90%  1.41x      93%  1.22x      94%  never

## What it settles

The pre-registered criterion (`ACTIVATION_PROTOCOL.md`) asked the homotopy arm
to beat the baseline on test accuracy. It loses by 1.19 points at 19 σ.

It also **reverses the one result that had favoured the homotopy**. On 10 000
images (exp3) the staircase won 17% of test loss and cut the generalisation
gap; here both flip sign:

| | 10k images (exp3) | 50k images (exp4) |
|---|---|---|
| accuracy | -0.96 % | **-1.19 %** (worse) |
| test loss | **-17 % (favourable)** | **+5.7 % (against)** |
| gen gap | -0.024 (favourable) | **+0.0075 (against)** |

So the loss win was "training less regularises", not a property of the
homotopy. At 10k the baseline memorises outright (gap +16.2 %) and anything
that slows learning helps the test loss; at 50k its gap is +5.2 %, there is no
overfitting left to correct, and the staircase -- which now overfits *more* --
has nothing to offer.

## Figures

`compare_updates.png` -- six panels against gradient updates. Thick line is the
mean over seeds, band is ±1 std, thin lines are the individual runs. Read the
top-left panel first: it is the only one where both arms are evaluated as the
same network (`test_acc_at_a0` for the staircase, BatchNorm re-estimated).

`alpha.png` -- per activation site, how far each nonlinearity actually is from
linear, against the alpha the schedule asked for. `linear_gap` rises as alpha
falls, so the homotopy was delivered rather than routed around by a BatchNorm
shift. This is the figure that says the negative result is about the method,
not about a broken implementation.

`loss_vs_alpha.png` -- L_alpha(theta_t) at fixed theta, swept over alpha, at
several checkpoints. Says whether the weights are good across the path or only
at the alpha they were trained at.

`stats.json` -- the resolved config, per-seed numbers and aggregate statistics.

## Caveat worth stating

`a_lr_restart` gives each plateau its own warmup and cosine, so over the final
22 epochs the staircase trains under a full cosine from lr=0.1 while the
baseline is in the tail of its own at lr<0.02. The comparison therefore carries
a schedule difference as well as a homotopy one.

Measured, it does not explain the gap: at epoch 57, before the restart, the
staircase was 1.78 points behind; at epoch 79 it was 1.27 behind. The final
phase *closed* half a point of relative ground despite the visible crash from
0.892 to 0.753. Removing the restart would likely have made the arm worse, not
better.

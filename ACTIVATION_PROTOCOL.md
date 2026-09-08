# Activation homotopy -- the protocol, written before the results

    phi_alpha(x) = max(x, alpha*x),   alpha : 1 -> 0

`alpha = 0` is exactly ReLU. `alpha = 1` is the identity, which makes the whole
ResNet affine. The question is whether introducing the nonlinearity gradually
makes the optimisation easier, and -- separately -- whether the *path* is what
helps or merely the starting point.

## What is different about this axis

Two facts distinguish it from the residual homotopy, and both change what the
results can mean.

**alpha cannot be reparametrised away.** `s * BN(.)` is the same function as BN
with `gamma -> s*gamma`, so a residual-homotopy run can absorb `s` and quietly
become a baseline. `phi_alpha` is positively homogeneous -- `phi_alpha(c*x) =
c*phi_alpha(x)` for `c > 0` -- so no rescaling of any weight changes the mix of
`x` and `|x|` it applies. This axis deforms the network, not its
parametrisation.

The escape that remains is **shift**, not scale: BatchNorm need only push beta
positive until nearly every pre-activation is positive, and `phi_alpha` is then
the identity for every alpha. `landscape.activation_stats` measures exactly
that:

    linear_gap = rms(phi_alpha(h) - h) / rms(h) = (1 - alpha) * rms(relu(-h)) / rms(h)

`linear_gap` must rise as alpha falls. Flat means the homotopy was routed
around, and that invalidates the run before any accuracy is read.

**alpha = 1 is degenerate, and more so than s = 0 was.** `s = 0` left a real
shallow ConvNet -- the stem, three 1x1 projections, the head. `alpha = 1` leaves
a linear classifier: its optimum on CIFAR-10 is around 40%, and its Hessian has
a large null space by construction, since a deep linear network is invariant
under `W1 -> W1*G, W2 -> G^-1*W2`. `theta*(1)` is a manifold, not an isolated
point, so the implicit function theorem has nothing to say there and a
continuation started exactly at alpha = 1 starts on a singular point of the
branch it is trying to follow. That is why `act_staircase` starts at 0.9 and
`act_half` at 0.5.

## Arms

All at 40 epochs, 3 seeds, `ckpt_every: 5`, `diag_every: 5`, everything except
the `a_*` keys identical to `baseline40.yaml`.

| Arm | Config | alpha(t) | Question |
|---|---|---|---|
| A | `baseline40` | 0 throughout | the reference |
| B | `act_linear` | 1 -> 0 linearly over [0, 0.5] | does the continuous path help? |
| C | `act_jump` | 1 until 0.3, then 0 | **the control**: or is it just a warm start? |
| D | `act_staircase` | 0.9, 0.6, 0.3, 0 with lr restarts | continuation: theta settles at each alpha_k |
| E | `act_stagewise` | sequential over 5 stage groups | nonlinear depth grows from the stem out |
| G | `act_half` | 0.5 -> 0 linearly | is the useful part of the path near alpha = 1 at all? |

**B versus C is the experiment.** Everything else contextualises it.

## Decision criterion, fixed now

> B is retained iff
> `test_acc_selected(B) - test_acc_selected(A) > 2 * combined sem`
> **and** `test_acc_selected(B) > test_acc_selected(C)` by the same margin.
>
> If B > A but B ~ C, the effect is an initialisation, not a continuation, and
> that is what gets written down.
>
> If `linear_gap` stays flat while alpha falls, the run is invalid before
> accuracy is read at all.

## Reading a run

`test_acc` during the ramp is the accuracy of whatever network is currently
running -- at alpha = 1 that is a linear classifier and the number means
nothing about a ResNet. `test_acc_at_a0` is the ReLU ResNet those weights
define, re-estimating BatchNorm first because dropping alpha to 0 removes the
entire negative mass of every activation and the stored statistics describe a
distribution the network no longer produces. Only `test_acc_at_a0` is
comparable to a baseline number.

    python explore.py runs/latest --only alpha,loss_vs_alpha,branch

    alpha.png          linear_gap and neg_frac per site over training, against
                       the alpha the schedule asked for. Check this first.
    loss_vs_alpha.png  L_alpha(theta_t) at fixed theta. Says whether the weights
                       are good across the path or only at the alpha they saw.
    branch.png         consecutive checkpoints compared as the networks they
                       define, each read at its own alpha. Parameter distance
                       against functional distance.

## The solution curve

`act_staircase` is the continuation arm: `a_lr_restart` gives every plateau its
own warmup and decay, so `theta` can settle at each `alpha_k` before the
continuation steps. `phase_bounds` reports the boundaries and
`grad_norm_total` at each one says whether the plateau actually settled.

Honest caveat: 8 epochs per plateau is not convergence for a ResNet-18. This
approximates continuation; it is not continuation. If the phase-boundary
gradient norms come out large, that is the finding, and the fix is more epochs
per phase rather than more plateaus.

`branch.png` is where continuity gets measured. Under BatchNorm the loss is
invariant to the scale of every conv weight, so `||theta_a - theta_b||` is
partly a gauge choice and functional distance is the honest measure. A step
that is small in parameters and large in function is where `theta*(alpha)`
stops being a branch -- a bifurcation candidate, and the result this axis is
most likely to produce.

**Not implemented, deliberately:** predictor-corrector continuation and any
tangent `dtheta/dalpha` requiring Hessian-vector products. The secant predictor
`theta*(alpha_k) + (theta*(alpha_k) - theta*(alpha_{k-1}))/(alpha_k -
alpha_{k-1}) * dalpha` is the cheap version to try first if the branch
measurements say the path is smooth.

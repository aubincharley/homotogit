"""The anchor penalty: L(theta, alpha) = L_CE(theta) + lambda * (alpha/2) * ||theta - theta_0||^2

Equation 12 of the GRDH paper, carried over to the activation homotopy.

The problem it solves is specific and it is not regularisation in the usual
sense. At alpha = 1 every activation is the identity, so the network is affine,
and a deep linear network is invariant under W1 -> W1*G, W2 -> G^-1*W2 for any
invertible G. Its Hessian therefore has a large null space *by construction*:
theta*(1) is a manifold, not an isolated point, and the implicit function
theorem -- the thing that promises a branch theta*(alpha) exists at all -- has
nothing to say there. A continuation started at alpha = 1 starts on a singular
point of the branch it is trying to follow.

Adding a convex quadratic centred on the random initialisation makes the
regularised Hessian nonsingular, which picks one point out of that manifold and
makes the start of the path well posed.

Two properties of the alpha coefficient matter:

  * it vanishes at alpha = 0, so the tail of training minimises the baseline
    objective exactly. A homotopy arm whose final epochs carried a penalty the
    baseline did not would not be comparable to it, and no amount of care
    elsewhere would fix that.
  * it is largest at alpha = 1, which is where the degeneracy is.

Scope: conv and linear weights only, never BatchNorm. Two reasons, and the
first is fatal on its own -- `zero_init_residual` starts every bn2 gamma at
zero, so anchoring BatchNorm would pin each residual branch at F == 0 and the
network could not leave the linear model however far alpha fell. The second is
that the scale symmetries the anchor exists to break live in the weight
matrices; BatchNorm's own scale invariance is a different object.
"""
import torch
from torch import nn

ANCHORED_MODULES = (nn.Conv2d, nn.Linear)


def anchored_parameters(model):
    """(name, parameter) for every weight the anchor applies to.

    Walked by module type rather than by name, so a renamed layer cannot
    silently drop out of the anchor, and a BatchNorm cannot silently fall into
    it.
    """
    out = []
    for module_name, module in model.named_modules():
        if not isinstance(module, ANCHORED_MODULES):
            continue
        for param_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                out.append((f"{module_name}.{param_name}" if module_name
                            else param_name, param))
    return out


def get_theta_0(model):
    """A frozen copy of the anchored weights, as {name: tensor}.

    clone().detach(), in that order: a detached view would alias the live
    weights, the anchor would track the very thing it is meant to hold still,
    and the penalty would read zero for the whole run.

    Left on whatever device the model is on. For a ResNet-18 that is ~45 MB of
    conv weights, which is cheaper than moving it across the bus every step.
    """
    return {name: param.detach().clone()
            for name, param in anchored_parameters(model)}


def anchor_penalty(model, theta_0):
    """sum_i ||w_i - w_i^0||^2, as a differentiable scalar.

    Not divided by anything. The 1/2 and the lambda live at the call site,
    where alpha is known -- keeping them out of here is what lets the same
    number be logged raw and compared across arms with different lambdas.
    """
    total = None
    for name, param in anchored_parameters(model):
        term = (param - theta_0[name]).pow(2).sum()
        total = term if total is None else total + term
    if total is None:
        raise ValueError("no anchored parameter in this model")
    return total


@torch.no_grad()
def anchor_distance(model, theta_0):
    """||theta - theta_0||_2 over the anchored weights, as a float.

    The logged counterpart of the penalty: a norm rather than a squared norm,
    so it is on the scale of the weights themselves and comparable to the
    relative step sizes explore.py reports.
    """
    total = 0.0
    for name, param in anchored_parameters(model):
        total += float((param - theta_0[name]).pow(2).sum())
    return total ** 0.5


def anchor_coefficient(alpha, lam):
    """lambda * alpha / 2 -- what multiplies the penalty in the loss.

    One function rather than an expression inlined in the training loop, so
    the invariant that decides whether an anchored arm is comparable to the
    baseline at all -- that this is exactly 0 once alpha reaches 0 -- is
    stated in one place and tested there.
    """
    return lam * alpha / 2.0


def calibrate_lambda(ce, penalty, alpha, target):
    """lambda such that lambda * (alpha/2) * penalty == target * ce.

    Calibrated after a short warmup, never at epoch 0: theta is still exactly
    theta_0 there, the penalty is identically zero, and there is nothing to
    scale against. Once set, lambda is fixed for the run and recorded in
    results.json -- a lambda that kept re-deriving itself would make two runs
    of the same config incomparable.

    Returns 0.0 rather than an infinity when the scale is degenerate (theta has
    not moved, or alpha is already 0). A division there would produce a lambda
    that only shows up as a NaN loss some minutes into the run.
    """
    scale = (alpha / 2.0) * penalty
    if scale <= 0.0 or ce <= 0.0:
        return 0.0
    return target * ce / scale

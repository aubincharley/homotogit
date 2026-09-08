"""The anchor penalty, and what it has to be true for.

    L(theta, alpha) = L_CE(theta) + lambda * (alpha/2) * ||theta - theta_0||^2

At alpha = 1 the network is affine, and a deep linear network is invariant
under W1 -> W1*G, W2 -> G^-1*W2. Its Hessian therefore has a large null space
by construction and theta*(1) is a manifold rather than an isolated point --
the implicit function theorem has nothing to say there, so a continuation
started at alpha=1 starts on a singular point of the branch it is following.
The quadratic term makes the regularised Hessian nonsingular; the alpha
coefficient makes it vanish exactly where it must, so the tail of training at
alpha=0 minimises the baseline objective and nothing else.

The claims pinned down here: the penalty is exactly zero at theta_0, exactly
zero at alpha=0 whatever lambda is, reaches only conv and linear weights, and
sends no gradient into theta_0.
"""
import pytest
import torch

from cifarbase.anchor import (anchor_coefficient, anchor_distance,
                              anchor_penalty, anchored_parameters,
                              calibrate_lambda, get_theta_0)
from cifarbase.config import load_config
from cifarbase.model import ARCHS, ResNet


def make_model(seed=0, width=16):
    torch.manual_seed(seed)
    return ResNet(ARCHS["resnet18"], num_classes=10, width=width,
                  zero_init_residual=True).eval()


# --------------------------------------------------------------------------
# scope: which parameters the anchor reaches
# --------------------------------------------------------------------------

def test_the_anchor_reaches_conv_and_linear_weights():
    model = make_model()
    names = {name for name, _ in anchored_parameters(model)}
    assert "conv1.weight" in names
    assert "layer1.0.conv1.weight" in names
    assert "fc.weight" in names and "fc.bias" in names


def test_the_anchor_leaves_batchnorm_alone():
    """zero_init_residual sets gamma_bn2 to 0, so theta_0 holds a zero there.
    Anchoring it would pin every residual branch off for the whole descent --
    the network could not leave the linear model even as alpha fell. The
    scale symmetries the anchor exists to break live in the weight matrices
    anyway, not in BatchNorm."""
    model = make_model()
    names = {name for name, _ in anchored_parameters(model)}
    bn_names = {f"{n}.{p}" for n, m in model.named_modules()
                if isinstance(m, torch.nn.BatchNorm2d) for p in ("weight", "bias")}
    assert bn_names                      # the model does have BatchNorm
    assert not (names & bn_names)


def test_every_anchored_parameter_gets_a_frozen_copy():
    model = make_model()
    theta_0 = get_theta_0(model)
    assert set(theta_0) == {name for name, _ in anchored_parameters(model)}
    assert not any(t.requires_grad for t in theta_0.values())


def test_theta_0_does_not_alias_the_live_weights():
    """A view rather than a clone would make the anchor track the weights it is
    supposed to hold still, and the penalty would be zero forever."""
    model = make_model()
    theta_0 = get_theta_0(model)
    with torch.no_grad():
        model.conv1.weight.add_(1.0)
    assert not torch.equal(theta_0["conv1.weight"], model.conv1.weight)


# --------------------------------------------------------------------------
# the penalty itself
# --------------------------------------------------------------------------

def test_the_penalty_is_exactly_zero_at_theta_0():
    model = make_model()
    assert float(anchor_penalty(model, get_theta_0(model)).detach()) == 0.0


def test_the_penalty_is_the_sum_of_squared_displacements():
    model = make_model()
    theta_0 = get_theta_0(model)
    with torch.no_grad():
        model.conv1.weight.add_(0.5)
        model.fc.bias.add_(2.0)
    expected = 0.25 * model.conv1.weight.numel() + 4.0 * model.fc.bias.numel()
    assert float(anchor_penalty(model, theta_0).detach()) == pytest.approx(expected, rel=1e-5)


def test_moving_batchnorm_does_not_move_the_penalty():
    model = make_model()
    theta_0 = get_theta_0(model)
    with torch.no_grad():
        model.bn1.weight.add_(3.0)
    assert float(anchor_penalty(model, theta_0).detach()) == 0.0


def test_the_penalty_gradient_is_twice_the_displacement():
    model = make_model()
    theta_0 = get_theta_0(model)
    with torch.no_grad():
        model.fc.bias.add_(0.3)
    model.zero_grad(set_to_none=True)
    anchor_penalty(model, theta_0).backward()
    assert torch.allclose(model.fc.bias.grad,
                          torch.full_like(model.fc.bias, 0.6), atol=1e-6)
    assert model.bn1.weight.grad is None        # BatchNorm is outside the anchor


def test_the_distance_is_the_euclidean_norm():
    model = make_model()
    theta_0 = get_theta_0(model)
    with torch.no_grad():
        model.fc.bias.add_(0.5)
    expected = (0.25 * model.fc.bias.numel()) ** 0.5
    assert anchor_distance(model, theta_0) == pytest.approx(expected, rel=1e-5)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def test_lambda_is_calibrated_to_the_requested_share_of_the_cross_entropy():
    """lambda * (alpha/2) * ||dtheta||^2 == target * CE, solved for lambda.

    Calibrated after a short warmup rather than at epoch 0, where theta is
    still exactly theta_0 and the penalty is identically zero -- there is
    nothing to scale against there.
    """
    lam = calibrate_lambda(ce=2.0, penalty=8.0, alpha=0.5, target=0.1)
    assert lam * (0.5 / 2) * 8.0 == pytest.approx(0.1 * 2.0)


def test_calibration_refuses_a_degenerate_scale():
    """theta still at theta_0, or alpha already 0: nothing to calibrate
    against, and a division would return an infinity that only shows up as a
    NaN loss twenty minutes into the run."""
    assert calibrate_lambda(ce=2.0, penalty=0.0, alpha=0.5, target=0.1) == 0.0
    assert calibrate_lambda(ce=2.0, penalty=8.0, alpha=0.0, target=0.1) == 0.0


# --------------------------------------------------------------------------
# the coefficient, as training applies it
# --------------------------------------------------------------------------

@pytest.mark.parametrize("alpha,lam,expected", [
    (1.0, 0.4, 0.2),        # lambda * alpha / 2
    (0.5, 0.4, 0.1),
    (0.0, 0.4, 0.0),        # the tail of training is the baseline objective
    (1.0, 0.0, 0.0),        # lambda 0 is the ablation arm
])
def test_the_coefficient_is_lambda_times_alpha_over_two(alpha, lam, expected):
    assert anchor_coefficient(alpha, lam) == pytest.approx(expected)


def test_the_coefficient_vanishes_at_alpha_zero_whatever_lambda_is():
    """The invariant that makes an anchored arm comparable to the baseline at
    all: once alpha reaches 0 the objective must be the baseline's, exactly.
    An arm whose final epochs carried a penalty the baseline did not is not
    measuring the same minimum."""
    assert all(anchor_coefficient(0.0, lam) == 0.0
               for lam in (0.0, 1e-4, 1.0, 1e6))


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def test_the_anchor_is_off_by_default():
    cfg = load_config([], default="baseline")
    assert cfg["anchor_lambda"] == 0.0 and cfg["anchor_target"] == 0.0
    assert 0.0 < cfg["anchor_calibrate_at"] < 1.0


def test_the_anchor_needs_the_activation_homotopy():
    """The coefficient is lambda*alpha/2, so with no alpha to scale it the
    penalty is identically zero: a config asking for an anchor without a
    homotopy has asked for a no-op."""
    with pytest.raises(SystemExit, match="a_schedule"):
        load_config(["--anchor-lambda", "0.01"], default="baseline")


def test_lambda_and_target_are_mutually_exclusive():
    with pytest.raises(SystemExit, match="anchor_target"):
        load_config(["--a-schedule", "linear", "--anchor-lambda", "0.01",
                     "--anchor-target", "0.1"], default="baseline")


def test_a_valid_anchored_arm_resolves():
    cfg = load_config(["--a-schedule", "linear", "--anchor-target", "0.1",
                       "--seeds", "0"], default="baseline")
    assert cfg["anchor_target"] == 0.1 and cfg["anchor_lambda"] == 0.0


def test_calibration_refuses_more_than_one_seed():
    """train_once runs once per seed, so a target would derive a different
    lambda for each -- three models optimising three slightly different
    objectives, with the difference hidden inside one results.json. Calibrate
    on one seed, pin the resolved lambda, then run the rest."""
    with pytest.raises(SystemExit, match="seed"):
        load_config(["--a-schedule", "linear", "--anchor-target", "0.1",
                     "--seeds", "0,1,2"], default="baseline")


def test_a_pinned_lambda_takes_any_number_of_seeds():
    cfg = load_config(["--a-schedule", "linear", "--anchor-lambda", "3.4e-4",
                       "--seeds", "0,1,2"], default="baseline")
    assert cfg["anchor_lambda"] == pytest.approx(3.4e-4)


def test_calibration_may_not_be_asked_for_at_the_very_start():
    """||theta - theta_0||^2 grows a thousandfold over the first ten epochs, so
    a ratio measured near step 0 produces a lambda that freezes the network by
    epoch 5. Measured: lambda=2.94 from step 50 gives a penalty 3656x the
    cross-entropy at epoch 5."""
    with pytest.raises(SystemExit, match="anchor_calibrate_at"):
        load_config(["--a-schedule", "linear", "--anchor-target", "0.1",
                     "--seeds", "0", "--anchor-calibrate-at", "0.0"],
                    default="baseline")

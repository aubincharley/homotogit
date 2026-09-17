"""The objective family, and the two properties the training loop depends on."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from continuation_core import losses
from continuation_core.config import ExperimentConfig, LossConfig


def _batch(n=64, c=10, scale=3.0, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, c) * scale, torch.randint(0, c, (n,))


@pytest.mark.parametrize("name", losses.LOSSES)
def test_every_loss_is_a_per_sample_mean(name):
    """The trainer rebuilds the batch gradient from microbatches of 32.

    ``(loss * n_micro / n_batch).backward()`` summed over microbatches equals the
    full-batch gradient only for objectives that average over samples
    independently.  A loss that coupled samples would train differently at a
    different microbatch size, silently.
    """
    z, y = _batch()
    f = losses.build(LossConfig(name=name))
    halves = [f(z[i:i + 32], y[i:i + 32]) for i in (0, 32)]
    assert torch.allclose(f(z, y), sum(halves) / 2, atol=1e-6)


@pytest.mark.parametrize("name", losses.LOSSES)
def test_every_loss_gives_a_finite_gradient_on_confident_wrong_logits(name):
    z, y = _batch(scale=30.0)
    z = z.requires_grad_(True)
    losses.build(LossConfig(name=name))(z, y).backward()
    assert torch.isfinite(z.grad).all()


def test_focal_at_gamma_zero_is_cross_entropy():
    z, y = _batch()
    f = losses.build(LossConfig(name="focal", gamma=0.0))
    assert torch.allclose(f(z, y), F.cross_entropy(z, y), atol=1e-6)


def test_square_takes_the_logits_and_its_gradient_never_saturates():
    """``square`` is the logit form, whose gradient is ``2(z - y)/C``.

    This is the whole reason to run it: cross-entropy's gradient is ``(p - y)``,
    bounded in ``[-1, 1]``, and the Brier score's passes through the softmax
    Jacobian and vanishes on a confident mistake.  The logit form does neither,
    so the test pins the exact gradient rather than a property of it.
    """
    z, y = _batch(scale=30.0)
    z = z.requires_grad_(True)
    losses.build(LossConfig(name="square"))(z, y).backward()
    t = F.one_hot(y, 10).to(z.dtype)
    assert torch.allclose(z.grad, 2 * (z.detach() - t) / 10 / y.numel(), atol=1e-6)


def test_brier_saturates_where_square_does_not():
    """The contrast that makes the logit form the one worth running."""
    c = 10
    for name, saturates in (("square", False), ("brier", True)):
        z = torch.zeros(1, c)
        z[0, 0] = 60.0                       # confident, and wrong
        z = z.requires_grad_(True)
        losses.build(LossConfig(name=name))(z, torch.tensor([1])).backward()
        big = z.grad.abs().max().item()
        assert (big < 1e-12) is saturates, (name, big)


def test_normalise_by_classes_moves_the_gradient_by_exactly_c():
    z, y = _batch()
    g = []
    for norm in (True, False):
        zz = z.clone().requires_grad_(True)
        losses.build(LossConfig(name="square", normalise_by_classes=norm))(zz, y).backward()
        g.append(zz.grad)
    assert torch.allclose(g[1], g[0] * 10, atol=1e-6)


def test_label_smoothing_raises_the_loss_of_a_perfect_prediction():
    y = torch.tensor([0, 1])
    z = torch.zeros(2, 10)
    z[0, 0] = z[1, 1] = 40.0
    ce = losses.build(LossConfig(name="cross_entropy"))(z, y)
    ls = losses.build(LossConfig(name="label_smoothing", label_smoothing=0.1))(z, y)
    assert ce < 1e-6 < ls


def test_an_unknown_loss_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="unknown loss"):
        losses.build(LossConfig(name="hinge"))


@pytest.mark.parametrize("bad", [{"name": "label_smoothing", "label_smoothing": 1.0},
                                 {"name": "focal", "gamma": -1.0}])
def test_out_of_range_parameters_are_refused(bad):
    with pytest.raises(ValueError):
        losses.build(LossConfig(**bad))


def test_the_default_config_still_selects_plain_cross_entropy():
    """Every recorded run must remain reproducible from its config."""
    z, y = _batch()
    cfg = ExperimentConfig()
    assert cfg.loss.name == "cross_entropy"
    assert torch.allclose(losses.build(cfg.loss)(z, y), F.cross_entropy(z, y))


def test_loss_survives_a_config_round_trip():
    cfg = ExperimentConfig()
    cfg.loss = LossConfig(name="square", normalise_by_classes=False)
    back = ExperimentConfig.from_dict(cfg.to_dict())
    assert back.loss == cfg.loss

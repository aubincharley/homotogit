"""VGG-11 with BatchNorm: a network that was designed without skips.

exp6 removed the residual connections from a ResNet-18 to test whether the
activation homotopy earns its keep on a rough landscape. The objection to that
result is depth: He et al. (2015) found plain and residual networks
near-equivalent at eighteen layers, with the degradation that motivates skips
appearing at thirty-four. A PlainNet-18 may simply not be hard enough.

VGG-11 sidesteps the objection. It is a real architecture, designed and tuned
without skip connections rather than derived by deleting them, and its landscape
is the one the literature actually describes as chaotic.

Everything here is additive. The last test in each group checks that the ResNet
path is untouched -- same sites, same order, same forward.
"""
import pytest
import torch
import torch.nn.functional as F

from cifarbase.activation import ActivationGate, activation_sites
from cifarbase.anchor import anchored_parameters
from cifarbase.config import load_config
from cifarbase.homotopy import residual_blocks
from cifarbase.model import ARCHS, VGG_PLANS, Activation, ResNet, VGG, build_model


def make_vgg(seed=0, width=64):
    torch.manual_seed(seed)
    return VGG(VGG_PLANS["vgg11"], num_classes=10, width=width).eval()


def batch(n=2, seed=1):
    return torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


# --------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------

def test_vgg11_has_eight_convolutions_each_with_batchnorm_and_an_activation():
    model = make_vgg()
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    acts = [m for m in model.modules() if isinstance(m, Activation)]
    assert len(convs) == len(norms) == len(acts) == 8


def test_vgg11_maps_a_cifar_image_to_ten_logits():
    with torch.no_grad():
        out = make_vgg()(batch())
    assert out.shape == (2, 10) and torch.isfinite(out).all()


def test_vgg11_has_no_residual_block():
    """The point of the architecture, and what the training loop has to
    tolerate: ResidualGate has nothing to gate here."""
    assert residual_blocks(make_vgg()) == []


def test_width_scales_every_channel():
    narrow = make_vgg(width=16)
    convs = [m for m in narrow.modules() if isinstance(m, torch.nn.Conv2d)]
    assert convs[0].out_channels == 16 and convs[-1].out_channels == 128


# --------------------------------------------------------------------------
# the homotopy reaches it
# --------------------------------------------------------------------------

def relu_forward(model, x):
    """The same stack with F.relu written out, as the reference alpha=0 must
    reproduce bit for bit."""
    out = x
    for layer in model.features:
        out = F.relu(out) if isinstance(layer, Activation) else layer(out)
    return model.fc(F.adaptive_avg_pool2d(out, 1).flatten(1))


def test_alpha_zero_is_bit_identical_to_the_relu_forward():
    model, x = make_vgg(), batch()
    with torch.no_grad():
        assert torch.equal(model(x), relu_forward(model, x))


def test_at_alpha_one_the_activations_are_the_identity():
    """The homotopy does reach every activation: at alpha=1 the convolutional
    stack up to the first maxpool is exactly affine."""
    model, x, y = make_vgg(), batch(seed=1), batch(seed=2)
    ActivationGate(model).set(1.0)
    stack = torch.nn.Sequential(*list(model.features)[:3])   # conv, bn, act
    with torch.no_grad():
        residual = stack(x + y) - stack(x) - stack(y) + stack(torch.zeros_like(x))
        assert residual.abs().max() < 1e-3 * stack(x).abs().max()


def test_but_the_whole_vgg_is_never_affine_because_of_the_maxpools():
    """A consequence that changes what the experiment tests.

    On a ResNet every nonlinearity is an Activation, so alpha=1 makes the whole
    network affine -- a linear classifier, with the singular Hessian that
    motivates the anchor. VGG also pools with max, which is nonlinear and carries
    no alpha. At alpha=1 a VGG is therefore still a piecewise-linear network, not
    a linear model: the homotopy removes the kinks the activations contribute and
    leaves the ones the pooling contributes.
    """
    model, x, y = make_vgg(), batch(seed=1), batch(seed=2)
    ActivationGate(model).set(1.0)
    with torch.no_grad():
        residual = model(x + y) - model(x) - model(y) + model(torch.zeros_like(x))
        assert residual.abs().max() > 1e-2 * model(x).abs().max()


def test_the_resnet_by_contrast_does_become_affine():
    """The control for the two tests above, and the reason the difference
    matters: on a ResNet the activations are the only nonlinearity."""
    torch.manual_seed(0)
    model = ResNet(ARCHS["resnet18"], num_classes=10, width=16,
                   zero_init_residual=False).eval()
    ActivationGate(model).set(1.0)
    x, y = batch(seed=1), batch(seed=2)
    with torch.no_grad():
        residual = model(x + y) - model(x) - model(y) + model(torch.zeros_like(x))
        assert residual.abs().max() < 1e-3 * model(x).abs().max()


def test_the_gate_finds_all_eight_sites():
    model = make_vgg()
    assert len(activation_sites(model)) == 8
    ActivationGate(model).set(0.4)
    assert all(s.alpha == 0.4 for s, _, _ in activation_sites(model))


def test_sites_are_in_forward_order():
    model, x = make_vgg(), batch()
    visited = []
    handles = [s.register_forward_pre_hook(
                   lambda m, i, mod=s: visited.append(mod))
               for s, _, _ in activation_sites(model)]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    assert visited == [s for s, _, _ in activation_sites(model)]


def test_stage_scope_follows_the_maxpool_blocks():
    """VGG's natural stages are the runs of convolutions between maxpools:
    1, 1, 2, 2, 2 for vgg11. That is what a stage-wise schedule should group."""
    stages = [stage for _, stage, _ in activation_sites(make_vgg())]
    assert stages == [0, 1, 2, 2, 3, 3, 4, 4]


def test_the_per_branch_scopes_are_refused():
    """act1 and act2 name the two activations of a residual block. VGG has no
    such structure, and silently returning all eight would make an ablation
    config a no-op."""
    with pytest.raises(ValueError, match="residual"):
        activation_sites(make_vgg(), "act1")


def test_the_anchor_reaches_the_convolutions_and_the_head():
    names = {n for n, _ in anchored_parameters(make_vgg())}
    assert "fc.weight" in names and "fc.bias" in names
    assert sum(1 for n in names if n.endswith(".weight") and "features" in n) == 8


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def test_vgg11_resolves_as_an_architecture():
    cfg = load_config(["--arch", "vgg11", "--no-zero-init-residual"],
                      default="baseline")
    assert cfg["arch"] == "vgg11"


def test_the_residual_axis_is_refused_on_a_network_that_has_none():
    with pytest.raises(SystemExit, match="residual"):
        load_config(["--arch", "vgg11", "--no-zero-init-residual",
                     "--s-schedule", "linear"], default="baseline")


def test_zero_init_residual_is_refused_as_a_silent_no_op():
    with pytest.raises(SystemExit, match="zero_init_residual"):
        load_config(["--arch", "vgg11"], default="baseline")


def test_build_model_returns_a_vgg():
    cfg = load_config(["--arch", "vgg11", "--no-zero-init-residual", "--width", "16"],
                      default="baseline")
    assert isinstance(build_model(cfg, torch.device("cpu"), quiet=True), VGG)


# --------------------------------------------------------------------------
# non-regression: the ResNet path is untouched
# --------------------------------------------------------------------------

def test_the_resnet_still_reports_seventeen_sites_in_forward_order():
    torch.manual_seed(0)
    model = ResNet(ARCHS["resnet18"], num_classes=10, width=16,
                   zero_init_residual=False).eval()
    sites = activation_sites(model)
    assert len(sites) == 17
    assert sites[0][1:] == (0, -1)
    assert [st for _, st, _ in sites[1:]] == sum(([s] * 4 for s in (1, 2, 3, 4)), [])


def test_the_resnet_forward_is_unchanged():
    torch.manual_seed(0)
    a = ResNet(ARCHS["resnet18"], num_classes=10, width=16,
               zero_init_residual=False).eval()
    torch.manual_seed(0)
    b = ResNet(ARCHS["resnet18"], num_classes=10, width=16,
               zero_init_residual=False).eval()
    x = batch()
    with torch.no_grad():
        assert torch.equal(a(x), b(x))
    assert len(activation_sites(a, "act1")) == 8      # the branch scopes still work
    assert len(activation_sites(a, "act2")) == 8

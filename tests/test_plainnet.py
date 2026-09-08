"""PlainNet-18: the same network with the `+ x` removed, and nothing else.

The hypothesis it exists to test: residual connections already convexify the
loss surface (Li et al. 2018), so an activation homotopy has little left to fix
on a ResNet. Strip the skips and the surface goes back to being chaotic --
which is the regime where starting from a near-linear network should matter.

For that comparison to mean anything, `use_residual` has to be the *only*
thing that moves. These tests pin that down: the block is exactly its own
convolution stack, no branch survives, and the same flag threaded through the
config reaches every block.
"""
import pytest
import torch

from cifarbase.config import load_config
from cifarbase.model import ARCHS, Activation, BasicBlock, ResNet


def block_pair(in_ch=16, out_ch=16, stride=1, seed=0):
    """The same block, built twice, residual and plain, from one seed."""
    torch.manual_seed(seed)
    residual = BasicBlock(in_ch, out_ch, stride).eval()
    torch.manual_seed(seed)
    plain = BasicBlock(in_ch, out_ch, stride, use_residual=False).eval()
    return residual, plain


def sequential_forward(block, x):
    """The convolution stack alone, written out. No branch, by construction."""
    out = block.act1(block.bn1(block.conv1(x)))
    return block.act2(block.bn2(block.conv2(out)))


# --------------------------------------------------------------------------
# the block
# --------------------------------------------------------------------------

def test_a_plain_block_is_exactly_its_own_convolution_stack():
    """The strongest statement of "no skip": the output is bit-identical to the
    stack computed without any branch. If a `+ x` survived anywhere, or a
    shortcut were added with a zero weight rather than removed, this fails."""
    _, plain = block_pair()
    x = torch.randn(4, 16, 8, 8)
    assert torch.equal(plain(x), sequential_forward(plain, x))


def test_a_residual_block_is_not():
    """The control. Without it the test above would pass on a block whose
    convolutions happen to output zero."""
    residual, _ = block_pair()
    x = torch.randn(4, 16, 8, 8)
    assert not torch.equal(residual(x), sequential_forward(residual, x))


def test_a_plain_block_carries_no_shortcut_module():
    """Not an Identity, not a disabled projection: absent. A shortcut left in
    place would take gradient, hold parameters and show up in the count."""
    residual, plain = block_pair(16, 32, stride=2)
    assert isinstance(residual.shortcut, torch.nn.Sequential)   # a projection
    assert not hasattr(plain, "shortcut")


def test_the_plain_block_drops_only_the_projection_parameters():
    """Channels and filters are untouched -- the only parameters that go are
    the 1x1 projection and its BatchNorm, which exist solely to make x
    addable."""
    residual, plain = block_pair(16, 32, stride=2)
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(residual) - n(plain) == n(residual.shortcut)


def test_the_input_reaches_the_output_only_through_the_convolutions():
    """The graph-level statement. Zeroing every convolution weight leaves a
    plain block with no path from x to its output at all, so the output stops
    depending on x. A surviving skip would carry it through."""
    _, plain = block_pair()
    with torch.no_grad():
        plain.conv1.weight.zero_()
    x = torch.randn(4, 16, 8, 8, requires_grad=True)
    plain(x).sum().backward()
    assert x.grad is not None and float(x.grad.abs().max()) == 0.0


# --------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------

def make(use_residual, zero_init_residual=False, seed=0, width=16):
    torch.manual_seed(seed)
    return ResNet(ARCHS["resnet18"], num_classes=10, width=width,
                  zero_init_residual=zero_init_residual,
                  use_residual=use_residual).eval()


def test_the_flag_reaches_every_block():
    plain = make(use_residual=False)
    blocks = [m for m in plain.modules() if isinstance(m, BasicBlock)]
    assert len(blocks) == 8
    assert not any(hasattr(b, "shortcut") for b in blocks)


def test_the_activation_homotopy_still_reaches_a_plain_net():
    """The whole point of the experiment is to run the existing homotopy on
    this model, so the seventeen activation sites have to survive the change."""
    plain = make(use_residual=False)
    assert len([m for m in plain.modules() if isinstance(m, Activation)]) == 17


def test_a_plain_net_still_computes_something():
    plain, x = make(use_residual=False), torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = plain(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all() and float(out.abs().max()) > 0


def test_the_residual_net_is_untouched_by_the_flag_existing():
    """The default has to be byte-identical to before use_residual existed, or
    every number already measured stops being comparable."""
    torch.manual_seed(0)
    reference = ResNet(ARCHS["resnet18"], num_classes=10, width=16,
                       zero_init_residual=False)
    x = torch.randn(2, 3, 32, 32)
    assert torch.equal(reference.eval()(x), make(use_residual=True)(x))


# --------------------------------------------------------------------------
# the trap: zero_init_residual kills a plain net outright
# --------------------------------------------------------------------------

def test_zero_init_residual_without_skips_produces_a_dead_network():
    """gamma_bn2 = 0 makes every block output exactly zero once there is no
    x to add back, so the whole network collapses to a constant. This is the
    demonstration; the config refuses the combination because of it."""
    dead, x = make(use_residual=False, zero_init_residual=True), torch.randn(2, 3, 32, 32)
    blocks = [m for m in dead.modules() if isinstance(m, BasicBlock)]
    with torch.no_grad():
        assert float(blocks[0](torch.randn(2, 16, 32, 32)).abs().max()) == 0.0
        assert torch.equal(dead(x), dead(torch.randn(2, 3, 32, 32)))   # input-independent


def test_the_config_refuses_that_combination():
    with pytest.raises(SystemExit, match="zero_init_residual"):
        load_config(["--no-use-residual"], default="baseline")


def test_a_plain_config_resolves():
    cfg = load_config(["--no-use-residual", "--no-zero-init-residual"],
                      default="baseline")
    assert cfg["use_residual"] is False and cfg["zero_init_residual"] is False

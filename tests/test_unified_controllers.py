"""Every unified-batch and ablation cell still builds, and its target path is plain.

Regression test for the constructor-order crash introduced when the
adaptive-continuation commits changed ``SiteController.__init__`` to validate
the schedule through ``q_for`` (see docs/AUDIT.md).  No suite test built an
``AblationController`` before, so the crash was not caught.
"""
import torch

from continuation.ablation_ops import build_from_cell
from continuation.config import ModelConfig
from continuation.models import build_model
from scripts.unified_manifest import build_configs


def _fresh():
    torch.manual_seed(0)
    return build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=0)


def test_every_unified_cell_builds_and_bypasses_exactly():
    x = torch.rand(2, 3, 32, 32)
    for cfg in build_configs():
        model = _fresh().eval()
        ref = _fresh().eval()
        ref.load_state_dict(model.state_dict())
        ctrl, handles = build_from_cell(model, {**cfg, "seed": 0, "cell_id": cfg["id"] + "__seed0"})
        ctrl.set_epoch(0)
        assert torch.isfinite(model(x)).all(), cfg["id"]
        ctrl.bypass_all = True
        ctrl.set_state(0.0, 32)
        with torch.no_grad():
            assert torch.equal(model(x), ref(x)), cfg["id"]
        for h in handles:
            h.remove()

"""Loss-landscape slice around a trained checkpoint, filtered vs unfiltered.

The question this answers is *not* "where did two arms land" -- no plain arm
ever saved weights.  It is the sharper one: on **one** set of weights and
**one** slice through parameter space, how do the two objectives differ?

    L_sigma(theta)  the objective actually being trained at that epoch
    L_0(theta)      the same weights with the filters bypassed (exact identity)

Both are evaluated on the same grid, the same directions and the same images,
so the only thing that changes between the two surfaces is the operator.

With ``--recalibrate-bn`` the BatchNorm running statistics are recomputed at
every grid point, separately for each objective, before that objective is
evaluated.  Without it the comparison is confounded: these checkpoints are
BatchNorm nets whose buffers were estimated under the *filtered* feature
distribution, so bypassing the filters evaluates the network with statistics
that belong to a different input distribution -- exactly the contamination
``continuation/models/resnet_gn.py`` cites as its reason for GroupNorm.
Calibration uses held-out *training* images, never the evaluation set, so no
statistics leak from the images the loss is read on.

Directions follow Li et al. (2018): two random directions, normalized
filter-wise to the anchor's own filter norms, so distance along an axis means
the same thing everywhere.  Only weight tensors with more than one dimension
are perturbed -- normalization scales/shifts and biases are left at the anchor,
as in that paper, since rescaling them is not comparable across layers.

    py scripts/loss_landscape_blur.py --grid 21 --n-images 512

Writes results/landscape/landscape_<tag>.npz with both surfaces, the grid, and
enough provenance (checkpoint digest, sigma, image indices) to redraw or audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from continuation.campaign_ops import SiteController, attach_sites  # noqa: E402
from continuation.config import DataConfig, ModelConfig  # noqa: E402
from continuation.data import build_dataset  # noqa: E402
from continuation.models import build_model  # noqa: E402
from continuation.pipeline import ChannelNormalizer  # noqa: E402

RUN = ("results/kaggle_outputs/signals-cal3-20260910-113759/"
       "campaign_job0_20260910-113823/Gplateau_cal3__seed0")
OUT = ROOT / "results" / "landscape"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def detect_arch(state):
    """The checkpoints do not record the architecture; find the one that fits."""
    for arch in ("resnet20_bn_cifar", "resnet20_gn", "resnet18_bn_cifar"):
        try:
            m = build_model(ModelConfig(arch=arch), 10, seed=0)
        except Exception:
            continue
        missing, unexpected = m.load_state_dict(state, strict=False)
        if not missing and not unexpected:
            return arch
    raise SystemExit("no architecture in the registry matches the checkpoint")


def filter_normalised_direction(anchor, generator):
    """A random direction scaled, per output filter, to the anchor's norms.

    Tensors of dimension <= 1 (norm scales and shifts, biases) get a zero
    direction: they are held at the anchor rather than perturbed.
    """
    d = {}
    for k, w in anchor.items():
        if w.dtype.is_floating_point and w.dim() > 1:
            r = torch.randn(w.shape, generator=generator, dtype=w.dtype)
            wf = w.reshape(w.shape[0], -1).norm(dim=1).clamp_min(1e-12)
            rf = r.reshape(r.shape[0], -1).norm(dim=1).clamp_min(1e-12)
            scale = (wf / rf).reshape(-1, *([1] * (w.dim() - 1)))
            d[k] = r * scale
        else:
            d[k] = torch.zeros_like(w) if w.dtype.is_floating_point else None
    return d


def batchnorm_modules(model):
    return [m for m in model.modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def recalibrate_bn(model, ctrl, bypass, images, norm, batch):
    """Re-estimate BN running statistics for the current weights and objective.

    ``momentum=None`` makes each layer accumulate a cumulative average over the
    calibration pass, so the result is the exact mean/var over those batches
    rather than an exponentially weighted trace.
    """
    bns = batchnorm_modules(model)
    if not bns:
        return False
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    ctrl.bypass_all = bypass
    model.train()
    with torch.no_grad():
        for s in range(0, len(images), batch):
            model(norm(images[s:s + batch].float() / 255.0))
    model.eval()
    for m, mo in zip(bns, saved):
        m.momentum = mo
    return True


def stratified(labels, per_class, num_classes=10, seed=0):
    g = np.random.default_rng(seed)
    lab = labels.numpy()
    picks = [g.permutation(np.flatnonzero(lab == c))[:per_class]
             for c in range(num_classes)]
    return np.sort(np.concatenate(picks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--epoch", type=int, default=12,
                    help="checkpoint epoch; needs sigma > 0 to show a difference")
    ap.add_argument("--grid", type=int, default=21)
    ap.add_argument("--span", type=float, default=1.0)
    ap.add_argument("--n-images", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--recalibrate-bn", action="store_true",
                    help="re-estimate BN statistics per grid point and per "
                         "objective, on held-out training images")
    ap.add_argument("--calib-images", type=int, default=500)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    run = ROOT / a.run
    ckpt = run / ("checkpoint_ep%02d.pt" % a.epoch)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    anchor = {k: v.clone() for k, v in ck["model_state"].items()}
    arch = detect_arch(anchor)

    levels = json.load(open(run / "summary.json"))["controller"]["levels"]
    sigma = float(levels[min(a.epoch, len(levels) - 1)])
    if sigma <= 0:
        raise SystemExit("sigma is 0 at epoch %d: both objectives coincide "
                         "exactly, so the comparison would be empty" % a.epoch)

    bundle = build_dataset(DataConfig(root=str(ROOT / "data"), download=False,
                                      num_val=5000))
    idx = stratified(bundle.test.labels, a.n_images // 10, seed=a.seed)
    x = bundle.test.images[idx]
    y = bundle.test.labels[idx]
    norm = ChannelNormalizer(bundle.mean, bundle.std)

    cal_x = None
    if a.recalibrate_bn:
        cidx = stratified(bundle.train.labels, a.calib_images // 10,
                          seed=a.seed + 1000)
        cal_x = bundle.train.images[cidx]

    model = build_model(ModelConfig(arch=arch), 10, seed=0)
    model.load_state_dict(anchor)
    model.eval()
    ctrl = SiteController("gaussian", levels=levels, resolution_by_epoch=[32] * 30,
                          reduction="input_bilinear")
    attach_sites(model, ctrl)
    ctrl.set_epoch(a.epoch)

    g = torch.Generator().manual_seed(a.seed)
    d1 = filter_normalised_direction(anchor, g)
    d2 = filter_normalised_direction(anchor, g)

    xs = np.linspace(-a.span, a.span, a.grid)
    ys = np.linspace(-a.span, a.span, a.grid)
    blur = np.full((a.grid, a.grid), np.nan)
    plain = np.full((a.grid, a.grid), np.nan)

    def evaluate(bypass):
        if cal_x is not None:
            recalibrate_bn(model, ctrl, bypass, cal_x, norm, a.batch)
        ctrl.bypass_all = bypass
        tot = 0.0
        with torch.no_grad():
            for s in range(0, len(x), a.batch):
                xb = norm(x[s:s + a.batch].float() / 255.0)
                tot += float(F.cross_entropy(model(xb), y[s:s + a.batch],
                                             reduction="sum"))
        return tot / len(x)

    t0 = time.perf_counter()
    for i, al in enumerate(xs):
        for j, be in enumerate(ys):
            with torch.no_grad():
                for k, w in anchor.items():
                    if d1[k] is None:
                        continue
                    model.state_dict()[k].copy_(w + al * d1[k] + be * d2[k])
            blur[j, i] = evaluate(False)
            plain[j, i] = evaluate(True)
        done = (i + 1) * a.grid
        el = time.perf_counter() - t0
        print("row %2d/%d  %5.1f%%  %6.1fs elapsed, ~%.0fs left"
              % (i + 1, a.grid, 100 * done / a.grid ** 2, el,
                 el * (a.grid ** 2 - done) / max(done, 1)), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    # the span must be in the name: two runs that differ only by span are
    # different experiments, and leaving it out let one overwrite the other
    tag = "%s_ep%02d_g%d_s%03d_n%d%s" % (
        run.name, a.epoch, a.grid, round(a.span * 100), len(x),
        "_bnrecal" if a.recalibrate_bn else "")
    path = OUT / ("landscape_%s.npz" % tag)
    np.savez_compressed(
        path, xs=xs, ys=ys, loss_blur=blur, loss_plain=plain,
        sigma=sigma, epoch=a.epoch, arch=arch, span=a.span,
        n_images=len(x), image_indices=idx, seed=a.seed,
        recalibrated_bn=bool(a.recalibrate_bn),
        calib_images=(0 if cal_x is None else len(cal_x)),
        checkpoint=str(ckpt.relative_to(ROOT)), checkpoint_sha256_16=digest(ckpt),
        centre_blur=blur[a.grid // 2, a.grid // 2],
        centre_plain=plain[a.grid // 2, a.grid // 2])
    print("sigma", sigma, "arch", arch)
    print("centre: blur %.4f  plain %.4f"
          % (blur[a.grid // 2, a.grid // 2], plain[a.grid // 2, a.grid // 2]))
    print("wrote", path)


if __name__ == "__main__":
    main()

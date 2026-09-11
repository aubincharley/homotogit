r"""Operators for the **anti-aliasing ablation** (placement, masks, constant sigma,
BlurPool) plus the two training-free diagnostics.

This module is **purely additive**: it imports from :mod:`continuation.campaign_ops`
and adds new behaviour beside it.  No existing operator, schedule or result is
modified, and a campaign cell that does not name an ablation builder takes exactly
the path it took before.

The hypothesis under test
-------------------------
**H (anti-aliasing).**  The internal Gaussian helps because it low-pass filters
*before a subsampling operation*, preventing spectrum folding.

**H' (annealed smoothness constraint).**  It helps because it constrains the
function class to be spatially smooth at every depth, a constraint later relaxed.

In ``resnet20_bn_cifar`` the subsampling operations are, exactly:

===============================  =========================================
``blocks[3].conv1``              ``stride=2``, site 7 of the 19 hooked convs
``blocks[3].shortcut``           ``x[:, :, ::2, ::2]``, never hooked
``blocks[6].conv1``              ``stride=2``, site 13
``blocks[6].shortcut``           ``x[:, :, ::2, ::2]``, never hooked
===============================  =========================================

Two facts follow, and they are what the ablation manipulates:

1. The hook used by the campaign fires on a convolution's **output**.  At sites 7
   and 13 the decimation has therefore *already happened*: the blur there cannot
   be anti-aliasing, it smooths an already-folded signal.
2. ``blocks[3]`` and ``blocks[6]`` each consume **one** input tensor, shared by
   their strided convolution and by their decimating shortcut.  A single
   ``forward_pre_hook`` on those two blocks therefore prefilters **all four**
   subsampling operations at once, in the textbook position -- after the previous
   block's ReLU, immediately before the decimation.

Placements
----------
``conv_out``    the campaign's placement: output of each of the 19 main-path 3x3
                convolutions, i.e. *before* BatchNorm and *before* ReLU.
``post_bn``     output of each of the 19 BatchNorm2d layers, still before ReLU.
                Same 19 positions, differing from ``conv_out`` only by the
                normalization -- but note BN then accumulates its running
                statistics on **unfiltered** activations, which ``conv_out`` does
                not.  That is a real coupled change, not a pure relocation.
``post_block``  after the ReLU: the stem's post-ReLU activation plus the 9 block
                outputs, i.e. **10** positions rather than 19.  This is the only
                placement in which the filter is anti-aliasing-correct at the two
                decimation points.  The site count differs by construction, so a
                ``post_block`` vs ``conv_out`` gap confounds placement with
                coverage; ``blurpool`` below is the clean version of the same idea.

Site masks (``conv_out`` indexing)
----------------------------------
``all19``    every site (campaign default).
``predown``  ``{6, 12}`` -- the conv outputs closest to a decimation.  They are
             *approximations* of the pre-decimation tensor: BN, the shortcut add
             and the ReLU still intervene between site 6 and ``blocks[3]``.
``nodown``   the other 17 sites.  Under H this should carry almost nothing.

Constant-sigma arms
-------------------
A constant level never reaches the target endpoint, so such an arm is **not a
continuation** -- it is a fixed architectural modification, and it is included
precisely as the control that separates "the filter helps" from "annealing the
filter helps".  Its honest final metric is the **current path** (filter active),
never the target path, which for a BatchNorm model measures a premature ablation
including running-statistic mismatch.  Cells declare this via ``primary_path``.

BlurPool
--------
A **fixed** Gaussian on the inputs of ``blocks[3]`` and ``blocks[6]``.  It is
architecture, not schedule: it is never annealed and it is deliberately **not**
disabled by ``bypass_all``, so the target path of a BlurPool arm is that
architecture evaluated at 32x32 without the annealed Gaussian -- which is its
genuine final configuration.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .campaign_ops import N_SITES, SiteController, reduce_spatial
from .transforms.gaussian import GaussianSmoothing

PLACEMENTS = ("conv_out", "post_bn", "post_block")

#: conv_out sites whose output is closest to a decimation (see module docstring)
PREDOWN = (6, 12)
NODOWN = tuple(s for s in range(N_SITES) if s not in PREDOWN)
#: post_block positions feeding a decimation: 0 = stem, 1..9 = block outputs
PREDOWN_POST_BLOCK = (3, 6)
#: blocks whose single input tensor feeds both a strided conv and a decimating shortcut
DECIMATING_BLOCKS = (3, 6)

MASKS = {"all19": tuple(range(N_SITES)), "predown": PREDOWN, "nodown": NODOWN}

#: Where a single resolution reduction is applied, and which ``post_block``
#: position is the first one to see the reduced grid.
#:
#: ``stem`` reduces the tensor entering ``model.blocks``; the reduction hook is
#: registered before the position-0 blur, so position 0 already sees the reduced
#: grid.  ``block{k}`` reduces the tensor entering ``model.blocks[k+1]``; the
#: forward hook carrying position ``k+1`` fires when ``blocks[k]`` *returns*,
#: i.e. before that pre-hook, so position ``k+1`` is still at full resolution.
#: Hence ``first_reduced = k + 2``.
INTERNAL_REDUCTIONS = {"stem": {"block_index": None, "first_reduced": 0}}
#: ``block{k}`` reduces the tensor entering ``model.blocks[k+1]``.  blocks[8] is
#: excluded: its output goes straight to global average pooling, so a reduction
#: there is consumed by the pooling rather than by any convolution.
for _k in range(8):
    INTERNAL_REDUCTIONS["block%d" % _k] = {"block_index": _k + 1,
                                           "first_reduced": _k + 2}


def relative_target(current: int, scheduled: int, base: int = 32) -> int:
    r"""Absolute schedule -> a target for a map of size ``current``.

    The campaign's schedule is written in absolute pixels for a 32x32 map
    (16 / 24 / 32).  Past ``blocks[3]`` the map is already 16x16 and past
    ``blocks[6]`` it is 8x8, so those absolute values stop meaning anything:
    16 becomes a no-op and 24 an *upsample*, which is not a resolution
    reduction at all.

    Read as a ratio instead -- ``f = scheduled / 32`` applied to the local size
    -- the schedule is well defined at every depth, and at a 32x32 insertion
    point it reproduces the absolute values exactly, so the runs already
    completed stay valid.
    """
    return max(1, int(round(current * float(scheduled) / float(base))))


def reduction_kind(reduction: str):
    """``('stem'|'block0'|..., 'max'|'bilinear')`` or ``None`` for input paths."""
    for name in INTERNAL_REDUCTIONS:
        if reduction.startswith(name):
            return name, reduction[len(name) + 1:]
    return None


class AblationController(SiteController):
    """A :class:`SiteController` that also knows where to sit and what stays fixed.

    Everything the campaign driver touches (``value``, ``resolution``, ``q``,
    ``bypass_all``, ``set_epoch``, ``set_state``, ``input_resolution``) is
    inherited unchanged, so the existing training loop and its two-path
    evaluation work without modification.
    """

    def __init__(self, operator="gaussian", levels=None, sites=None,
                 resolution_by_epoch=None, reduction="input_bilinear",
                 placement="conv_out", blurpool_sigma=None,
                 n_positions=None, mask_name="all19", constant=False,
                 relative=False, sigma_profile=None):
        if placement not in PLACEMENTS:
            raise ValueError("unknown placement %r; known: %s" % (placement, PLACEMENTS))
        super().__init__(operator=operator, levels=levels, sites=sites,
                         resolution_by_epoch=resolution_by_epoch,
                         reduction=reduction)
        self.placement = placement
        #: read resolution_by_epoch as a ratio of 32 rather than as pixels
        self.relative = bool(relative)
        #: Per-position multiplier on sigma -- the depth prior.  It multiplies
        #: **on top of** ``q``, which is deliberate: with the profile computed on
        #: the network's natural map sizes, ``m_l * q_l`` makes sigma track the
        #: *current* map size, including during the reduced phase.  ``None``
        #: leaves every position at 1.0, i.e. the uniform sigma studied so far.
        self.sigma_profile = ([float(v) for v in sigma_profile]
                              if sigma_profile else None)
        self.mask_name = mask_name
        self.constant = bool(constant)
        self.n_positions = int(n_positions) if n_positions else N_SITES
        # BlurPool is architecture: fixed width, never annealed, never bypassed.
        self.blurpool_sigma = None if blurpool_sigma is None else float(blurpool_sigma)
        self.blurpool = (GaussianSmoothing(sigma_max=max(self.blurpool_sigma, 1e-6),
                                           truncate=4.0)
                         if self.blurpool_sigma else None)
        # post_block indexes 10 positions, so the inherited per-site q table
        # (sized N_SITES) must not be indexed past its end.
        self.q = [1.0] * max(self.n_positions, N_SITES)

    def internal_reduction(self):
        """``(name, spec)`` when the reduction happens inside the network."""
        k = reduction_kind(self.reduction)
        return (k[0], INTERNAL_REDUCTIONS[k[0]]) if k else None

    def input_resolution(self):
        """Any internal reduction leaves the network input untouched."""
        if self.resolution is None:
            return None
        return None if self.internal_reduction() else self.resolution

    def stem_resolution(self):
        """Kept for the ``stem`` path; deeper points use their own hook."""
        if self.resolution is None:
            return None
        k = self.internal_reduction()
        return self.resolution if (k and k[0] == "stem") else None

    def q_for(self, r):
        if self.placement == "post_block":
            # every post_block position sits after the stem reduction, so they
            # all see the reduced grid -- including position 0
            ratio = 1.0 if r is None else float(r) / 32.0
            k = self.internal_reduction()
            first = k[1]["first_reduced"] if k else 0
            n = max(self.n_positions, N_SITES)
            # positions upstream of the reduction still see the full grid
            return [1.0 if i < first else ratio for i in range(n)]
        base = super().q_for(r)
        if self.n_positions > len(base):
            base = base + [base[-1]] * (self.n_positions - len(base))
        return base

    def apply_at(self, site: int, h: torch.Tensor) -> torch.Tensor:
        """As :class:`SiteController`, with the depth prior folded into sigma.

        The profile multiplies **on top of** ``q``.  Overriding rather than
        editing the parent keeps ``campaign_ops`` untouched, so every historical
        arm takes exactly the path it took before.
        """
        if self.sigma_profile is None:
            return super().apply_at(site, h)
        if not self.is_active() or site not in self.sites:
            return h                                   # exact bypass
        q = self.q[site] * self.sigma_profile[site]
        if self.operator == "gaussian":
            return self.gauss(h, q * float(self.value))
        alpha = float(self.value)                      # gmix, reference width 1
        return (1.0 - alpha) * h + alpha * self.gauss(h, q)

    def apply_blurpool(self, h):
        """Fixed anti-aliasing prefilter.  Independent of ``bypass_all``."""
        if self.blurpool is None:
            return h
        return self.blurpool(h, self.blurpool_sigma)

    def describe(self):
        d = super().describe()
        d.update(placement=self.placement, mask=self.mask_name,
                 n_positions=self.n_positions,
                 constant_level=self.constant,
                 blurpool_sigma=self.blurpool_sigma,
                 relative=self.relative, sigma_profile=self.sigma_profile,
                 sigma_profile_note=("per-position multiplier on sigma, applied on "
                                     "top of q; None = uniform sigma"),
                 reduction_point=(None if not self.internal_reduction()
                                  else self.internal_reduction()[0]),
                 first_reduced_position=(None if not self.internal_reduction()
                                         else self.internal_reduction()[1]["first_reduced"]),
                 blurpool_note=("fixed architectural prefilter on the inputs of "
                                "blocks[3] and blocks[6]; covers the strided conv "
                                "and the decimating shortcut; not disabled by "
                                "bypass_all"),
                 placement_note={
                     "conv_out": "conv output, before BN and ReLU (campaign default)",
                     "post_bn": "BatchNorm output, before ReLU; BN statistics are "
                                "accumulated on unfiltered activations",
                     "post_block": "post-ReLU: stem activation + 9 block outputs "
                                   "(10 positions, not 19)",
                 }[self.placement])
        return d


# --------------------------------------------------------------------------
# attachment
# --------------------------------------------------------------------------

def _conv_sites(model):
    return [m for m in model.modules()
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3)]


def _bn_sites(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


def attach_ablation(model, ctrl: AblationController):
    """Hook ``model`` according to ``ctrl.placement`` and its BlurPool setting.

    Returns the handle list, exactly like ``campaign_ops.attach_sites``.
    """
    handles = []

    # Resolution reduction after the stem's BN+ReLU, when the arm asks for one.
    # Registered FIRST so that, for post_block, position 0 blurs the tensor that
    # has already been reduced -- reduce, then blur, declared rather than left to
    # hook registration order.  A no-op for every arm whose reduction is applied
    # to the input image instead.
    kind = ctrl.internal_reduction()
    if kind is not None:
        name, spec = kind
        how = ctrl.reduction[len(name) + 1:]

        def _reduce(_mod, args):
            r = ctrl.resolution
            if r is None:
                return None
            if ctrl.relative:
                r = relative_target(int(args[0].shape[-1]), r)
            return (reduce_spatial(args[0], r, how),)

        target = (model.blocks if spec["block_index"] is None
                  else model.blocks[spec["block_index"]])
        handles.append(target.register_forward_pre_hook(_reduce))

    if ctrl.placement == "conv_out":
        mods = _conv_sites(model)
        if len(mods) != N_SITES:
            raise RuntimeError("expected %d conv sites, found %d" % (N_SITES, len(mods)))
        for i, m in enumerate(mods):
            handles.append(m.register_forward_hook(
                lambda _m, _i, out, s=i: ctrl.apply_at(s, out)))

    elif ctrl.placement == "post_bn":
        mods = _bn_sites(model)
        if len(mods) != N_SITES:
            raise RuntimeError("expected %d BN sites, found %d" % (N_SITES, len(mods)))
        for i, m in enumerate(mods):
            handles.append(m.register_forward_hook(
                lambda _m, _i, out, s=i: ctrl.apply_at(s, out)))

    elif ctrl.placement == "post_block":
        # position 0: the stem's post-ReLU activation, i.e. the input of blocks
        def _stem(_mod, args):
            return (ctrl.apply_at(0, args[0]),)
        handles.append(model.blocks.register_forward_pre_hook(_stem))
        for b, block in enumerate(model.blocks):
            handles.append(block.register_forward_hook(
                lambda _m, _i, out, s=b + 1: ctrl.apply_at(s, out)))

    # BlurPool sits on the input of each decimating block, before every hook the
    # block itself carries, and independently of the annealed operator above.
    if ctrl.blurpool is not None:
        for b in DECIMATING_BLOCKS:
            def _pre(_mod, args):
                return (ctrl.apply_blurpool(args[0]),)
            handles.append(model.blocks[b].register_forward_pre_hook(_pre))

    return handles


def build_from_cell(model, cell):
    """Cell dict -> ``(controller, handles)``.  The ablation's entry point.

    Named by ``cell["controller_builder"]`` so that
    ``scripts/campaign_driver.py`` can dispatch to it without any ablation
    knowledge of its own.
    """
    from scripts.ablation_manifest import LEVELS, RESOLUTIONS_ABL
    try:                      # wave 2 adds Rprog; wave 1 cells are unaffected
        from scripts.ablation2_manifest import RESOLUTIONS_ABL2
        RESOLUTIONS_ABL = {**RESOLUTIONS_ABL, **RESOLUTIONS_ABL2}
    except ImportError:
        pass

    placement = cell.get("placement", "conv_out")
    mask_name = cell.get("mask", "all19")
    if placement == "post_block":
        n_positions, sites = 10, tuple(range(10))
    else:
        n_positions, sites = N_SITES, MASKS[mask_name]

    levels = LEVELS[cell["levels"]]
    ctrl = AblationController(
        relative=bool(cell.get("relative_reduction", False)),
        sigma_profile=cell.get("sigma_profile"),
        operator=cell["operator"], levels=levels, sites=sites,
        resolution_by_epoch=RESOLUTIONS_ABL[cell["resolution"]],
        reduction=cell.get("reduction", "input_bilinear"),
        placement=placement, blurpool_sigma=cell.get("blurpool_sigma"),
        n_positions=n_positions, mask_name=mask_name,
        constant=bool(cell.get("constant", False)))
    return ctrl, attach_ablation(model, ctrl)


# --------------------------------------------------------------------------
# diagnostics 0a / 0b -- measured on trained weights, no extra training
# --------------------------------------------------------------------------

def _translate(x, dy, dx):
    """Shift by whole pixels with replicate padding (no wrap-around)."""
    pad = max(abs(dy), abs(dx))
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate")
    h, w = x.shape[-2:]
    top, left = pad - dy, pad - dx
    return xp[..., top:top + h, left:left + w]


@torch.no_grad()
def shift_consistency(model, pipe, images, labels, ctrl=None, res=None,
                      shifts=((0, 1), (1, 0), (1, 1)), batch=500):
    r"""Diagnostic **0b**: top-1 stability under a whole-pixel input translation.

    Anti-aliasing has a measurable signature -- it makes a network's prediction
    more stable under small input shifts (Zhang, *Making Convolutional Networks
    Shift-Invariant Again*, 2019).  If an arm's benefit comes from anti-aliasing,
    it should be **more shift-consistent** than its plain control.

    Reports, per shift and pooled: ``consistency`` (fraction of images whose
    arg-max is unchanged) and ``acc_shifted``.  ``acc_unshifted`` is reported
    once for reference.  This measures the network **as configured at call
    time** -- the caller decides whether internal filters are active.
    """
    was = model.training
    model.eval()
    n = int(images.shape[0])

    def logits_of(x_u8, dy, dx):
        outs = []
        for i in range(0, n, batch):
            xb = x_u8[i:i + batch]
            z = pipe(xb, 0.0, res=res)
            if (dy, dx) != (0, 0):
                z = _translate(z, dy, dx)
            outs.append(model(z).argmax(1))
        return torch.cat(outs)

    base = logits_of(images, 0, 0)
    acc0 = float((base == labels).float().mean())
    per_shift = {}
    cons_all, acc_all = [], []
    for (dy, dx) in shifts:
        pred = logits_of(images, dy, dx)
        cons = float((pred == base).float().mean())
        acc = float((pred == labels).float().mean())
        per_shift["dy%+d_dx%+d" % (dy, dx)] = {"consistency": cons, "acc_shifted": acc}
        cons_all.append(cons)
        acc_all.append(acc)
    if was:
        model.train()
    return {
        "n_images": n,
        "acc_unshifted": acc0,
        "consistency_mean": sum(cons_all) / len(cons_all),
        "acc_shifted_mean": sum(acc_all) / len(acc_all),
        "acc_drop_from_shift": acc0 - sum(acc_all) / len(acc_all),
        "per_shift": per_shift,
        "shift_method": "replicate padding then crop; no wrap-around",
        "note": ("translation is applied to the normalized network input, after "
                 "the pipeline, so it is a pure geometric shift of the tensor "
                 "the network sees"),
    }


@torch.no_grad()
def aliasing_energy(model, pipe, images, res=None, batch=500, max_batches=4):
    r"""Diagnostic **0a**: how much spectrum would actually fold at each decimation.

    Captures the input tensor of every decimating block and reports the fraction
    of squared-magnitude 2-D DFT energy lying **above the post-decimation Nyquist
    frequency** (``|f| > 0.25`` cycles/sample on either axis, for a factor-2
    decimation).  That fraction is the energy that aliases when the block
    subsamples.

    If it is small, there is little aliasing for a prefilter to prevent, and H is
    weak before any training comparison is made.  DC is included in the total.
    """
    was = model.training
    model.eval()
    captured = {}

    def make_hook(name):
        def _pre(_mod, args):
            captured[name] = args[0].detach()
            return None
        return _pre

    handles = [model.blocks[b].register_forward_pre_hook(make_hook("blocks%d" % b))
               for b in DECIMATING_BLOCKS]

    acc = {("blocks%d" % b): [] for b in DECIMATING_BLOCKS}
    shapes = {}
    n = int(images.shape[0])
    for bi, i in enumerate(range(0, n, batch)):
        if bi >= max_batches:
            break
        model(pipe(images[i:i + batch], 0.0, res=res))
        for name, t in captured.items():
            h, w = t.shape[-2:]
            shapes[name] = tuple(t.shape[1:])
            f = torch.fft.fft2(t.to(torch.float32), norm="ortho")
            p = (f.real ** 2 + f.imag ** 2)
            fy = torch.fft.fftfreq(h, device=t.device).abs().view(-1, 1)
            fx = torch.fft.fftfreq(w, device=t.device).abs().view(1, -1)
            above = ((fy > 0.25) | (fx > 0.25)).to(p.dtype)
            tot = p.sum(dim=(-2, -1))
            hi = (p * above).sum(dim=(-2, -1))
            acc[name].append(float((hi / tot.clamp_min(1e-12)).mean()))

    for h in handles:
        h.remove()
    if was:
        model.train()
    return {
        "definition": ("fraction of |DFT|^2 energy with |f_y|>0.25 or |f_x|>0.25 "
                       "cycles/sample, at the input of each decimating block; "
                       "that is the energy folded by a factor-2 subsampling"),
        "measured_at": {b: {"shape_CHW": shapes.get(b), "n_batches": len(v),
                            "alias_energy_fraction": (sum(v) / len(v)) if v else None}
                        for b, v in acc.items()},
        "note": ("measured on the network as configured at call time; the caller "
                 "decides whether internal filters are active"),
    }

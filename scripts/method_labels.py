"""Plain-English names for every configuration, readable without the codebase.

Vocabulary, used consistently:

* **blur**   -- the internal Gaussian applied to activations, annealed to zero
                unless the name says "fixed"
* **shrink** -- a real spatial resolution reduction, restored to 32x32 later
                unless the name says "never restored"
* positions  -- "the input image", "the first conv layer" (the stem), or
                "block 0 / 1 / 2" (after that residual block's output)
* schedules  -- written as the resolutions themselves, e.g. "16 to 24 to 32"

Anything that is a control rather than a candidate says so in the name.
"""
from __future__ import annotations

PATH_NAME = {
    "Rprog": "16 to 24 to 32",
    "Rgentle": "24 to 32 only",
    "Rreverse": "24 to 16 to 32 (order control)",
    "Rlate": "16 to 24 to 32, longer at low res",
    "fixed16": "fixed 16x16, never restored",
    "fixed24": "fixed 24x24, never restored",
    "none": "",
}
RES_NAME = {"R32": "", "Rprog": "16 to 24 to 32",
            "Rgentle": "24 to 32 only", "Rreverse": "24 to 16 to 32 (order control)"}
OPERATOR_NAME = {
    "max": "max-pool",
    "bilinear": "bilinear",
    "maxblur": "max-pool then anti-alias blur",
    "softpool": "softmax-weighted pooling",
    "l2": "least-squares reconstruction",
    "hminus1": "smoothness-optimal reconstruction",
    "perceptual": "perceptual (SSIM-style)",
}
WHERE_NAME = {"input": "the input image", "stem": "the first conv layer",
              "D0": "block 0", "D1": "block 1", "D2": "block 2"}
REDUCTION_NAME = {"input_bilinear": "the input image (bilinear)",
                  "input_max": "the input image (max-pool)",
                  "stem_bilinear": "the first conv layer (bilinear)",
                  "stem_max": "the first conv layer (max-pool)"}
BLUR_NAME = {"Gplateau": "blur every conv layer, annealed to zero",
             "Ggeo": "blur every conv layer, geometric decay",
             "Gmix": "blend each layer between itself and its blur"}


def campaign_label(cid: str) -> tuple:
    res, g, red, mask = cid.split("__")
    sites = "" if mask == "all19" else ", first 7 layers only"
    if g == "Gnone" and res == "R32":
        return "Baseline: no blur, full 32x32 throughout", "plain"
    if g == "Gnone":
        return ("Shrink %s, %s" % (REDUCTION_NAME[red], RES_NAME[res])), "res"
    if res == "R32":
        fam = "mix" if g == "Gmix" else "gauss"
        return ("%s%s" % (BLUR_NAME[g][0].upper() + BLUR_NAME[g][1:], sites)), fam
    return ("Shrink %s, %s, plus %s%s"
            % (REDUCTION_NAME[red], RES_NAME[res], BLUR_NAME[g], sites)), "res_gauss"


def resbench_label(cid: str) -> tuple:
    op, loc, path = cid.split("__")
    if op == "none":
        return "Baseline: no blur, full 32x32 throughout (resolution batch)", "plain"
    if path in ("fixed16", "fixed24"):
        return ("Shrink %s with %s, %s"
                % (WHERE_NAME[loc], OPERATOR_NAME[op], PATH_NAME[path])), "control"
    fam = "operator" if loc in ("input", "D1") and path == "Rprog" else "res"
    return ("Shrink %s with %s, %s"
            % (WHERE_NAME[loc], OPERATOR_NAME[op], PATH_NAME[path])), fam


ABLATION_LABEL = {
    "C_plain": ("Baseline: no blur, full 32x32 throughout (ablation batch)", "plain"),
    "C_plateau": ("Blur every conv layer, annealed to zero", "gauss"),
    "P_postbn": ("Blur after every BatchNorm, annealed to zero", "gauss"),
    "P_postblock": ("Blur after every ReLU (10 places), annealed to zero", "gauss"),
    "M_predown": ("Blur only the 2 layers feeding a downsample", "gauss"),
    "M_nodown": ("Blur the 17 layers away from a downsample", "gauss"),
    "K_const030": ("Fixed blur 0.30, never annealed (control)", "constant"),
    "K_const050": ("Fixed blur 0.50, never annealed (control)", "constant"),
    "K_const080": ("Fixed blur 0.80, never annealed (control)", "constant"),
    "K_const100": ("Fixed blur 1.00, never annealed (control)", "constant"),
    "B_blurpool": ("Fixed anti-alias blur before each downsample (control)",
                   "constant"),
    "B_blurpool_plateau": ("Fixed anti-alias blur before each downsample, "
                           "plus annealed blur everywhere", "res_gauss"),
    "R1": ("Shrink the input image, 16 to 24 to 32", "res"),
    "R2": ("Shrink the input image, 16 to 24 to 32, plus blur every conv layer",
           "res_gauss"),
    "R3": ("Shrink the input image, 16 to 24 to 32, plus blur after every ReLU",
           "res_gauss"),
    "R4": ("Shrink the first conv layer, 16 to 24 to 32", "res"),
    "R5": ("Shrink after the first ReLU, plus blur after every ReLU", "res_gauss"),
    "R6": ("Fixed blur after every ReLU, never annealed (control)", "constant"),
    "D0": ("Shrink block 0 with max-pool, 16 to 24 to 32", "res"),
    "D1": ("Shrink block 1 with max-pool, 16 to 24 to 32", "res"),
    "D2": ("Shrink block 2 with max-pool, 16 to 24 to 32", "res"),
    "D0G": ("Shrink block 0, plus blur after every ReLU", "res_gauss"),
    "D1G": ("Shrink block 1, plus blur after every ReLU", "res_gauss"),
    "D2G": ("Shrink block 2, plus blur after every ReLU", "res_gauss"),
    "P_A1": ("Blur strength scaled by map size, after every ReLU", "profile"),
    "P_A2": ("Blur strength scaled by sqrt of map size, after every ReLU",
             "profile"),
    "P_A3": ("Blur strength rising with depth, after every ReLU (direction control)",
             "profile"),
    "P_A4": ("Blur strength scaled by receptive field, after every ReLU", "profile"),
    "Q_A1": ("Blur strength scaled by map size, after every conv", "profile"),
    "Q_A2": ("Blur strength scaled by sqrt of map size, after every conv",
             "profile"),
    "Q_A3": ("Blur strength rising with depth, after every conv (direction control)",
             "profile"),
    "Q_A4": ("Blur strength scaled by receptive field, after every conv", "profile"),
}

ADAPTIVE_LABEL = {
    "ADAPT": ("Blur schedule chosen by the network (first trigger)", "adaptive"),
    "ADAPTSIG": ("Blur schedule chosen by a gradient-norm trigger", "adaptive"),
    "ADAPTGAP": ("Blur schedule chosen by a transfer-gap trigger (mis-set)",
                 "adaptive"),
    "ADAPTGAP2": ("Blur schedule chosen by a transfer-gap trigger", "adaptive"),
    "ADAPTRES": ("Resolution schedule chosen by a transfer-gap trigger", "adaptive"),
    "alloc_back": ("Blur schedule, more time at low blur", "adaptive"),
    "alloc_front": ("Blur schedule, more time at high blur", "adaptive"),
    "alloc_even": ("Blur schedule, even time at each level", "adaptive"),
    "Gplateau_cal": ("Blur every conv layer, annealed (instrumented rerun)", "gauss"),
    "Gplateau_cal2": ("Blur every conv layer, annealed (instrumented rerun 2)",
                      "gauss"),
    "Gplateau_cal3": ("Blur every conv layer, annealed (instrumented rerun 3)",
                      "gauss"),
    "rho0.5": ("Per-layer blur strength, exponent 0.5", "profile"),
    "rho1": ("Per-layer blur strength, exponent 1", "profile"),
    "rho2": ("Per-layer blur strength, exponent 2", "profile"),
    "adaptive": ("Per-layer blur strength chosen by the network", "profile"),
    "plain": ("Baseline: no blur, full 32x32 throughout (per-layer batch)", "plain"),
}

#: the curves worth drawing -- one per idea, not one per cell
REPRESENTATIVE = [
    "campaign/R32__Gnone__input_bilinear__all19",
    "campaign/R32__Gplateau__input_bilinear__all19",
    "campaign/Rprog__Gnone__input_bilinear__all19",
    "campaign/Rprog__Gplateau__input_bilinear__all19",
    "resbench/max__D1__Rprog",
    "resbench/max__D1__fixed16",
    "ablation/P_postblock",
    "ablation/D2G",
    "ablation/P_A4",
    "ablation/K_const050",
    "adaptive/ADAPTGAP2@30",
    "campaign/R32__Gmix__input_bilinear__all19",
]

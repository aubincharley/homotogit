"""Frozen manifest for the **anti-aliasing ablation**: 12 configurations x 1 seed.

Purely additive: it defines its own cells and its own pinned assets and does not
read, rewrite or reinterpret anything under ``results/kaggle_outputs/``.

Design
------
Every cell runs at **constant 32x32** (``R32``) so that the resolution question is
held fixed and the only manipulated variables are those that separate H
(anti-aliasing) from H' (annealed smoothness constraint) -- see
``continuation/ablation_ops.py`` for the hypothesis statement and for what each
placement and mask means physically.

The two controls are run **fresh, in this same batch**.  They are not optional:
the historical 63-cell grid used initial weights that are unrecoverable (the
Kaggle dataset carrying them belongs to accounts not configured here), so no cell
below may be compared to ``results/campaign_results.json``.  Within this batch,
every arm shares initial weights, BN buffers, per-epoch permutations and probe
indices, so all comparisons here are exactly paired.

One seed, deliberately
----------------------
``SEEDS = (0,)``: each configuration is run **once**.  What this design still
buys, and what it does not:

* it **does** buy pairing -- every arm starts from the same initial weights, the
  same BN buffers and the same per-epoch permutations, so a difference between
  two arms is not a difference of initialization or of batch order;
* it does **not** buy any estimate of run-to-run dispersion.  No SD, no
  "significant", no ranking of arms separated by a small margin.  Errata C-12 and
  C-13 apply directly: a single run defines no distribution, and a prior campaign
  measured two identical plain configurations differing by 0.04 pp on final
  accuracy -- one observed difference, not a noise floor.

Read the outcome as a **sign and an order of magnitude**.  The comparisons this
ablation is built to make (does the effect survive a placement change? does it
concentrate on the decimation sites? does it survive removing the annealing?) are
expected to be several points wide, which is why one seed is informative here;
any comparison that lands within a point should be reported as undecided and
re-run with more seeds before it is believed.

Reading the results
-------------------
``primary_path`` says which of the driver's two evaluation paths is the honest
final metric for that arm:

``target``   the target path (32x32, annealed filter bypassed).  Correct for
             every arm that ends on the target objective.
``current``  the current path (filter active).  Correct for the constant-sigma
             arms, which never reach the target endpoint: for them the target
             path measures a premature ablation with BatchNorm running-statistic
             mismatch, not predictor quality.
"""
from __future__ import annotations

EPOCHS, BYPASS_FROM, SEEDS = 30, 21, (0,)

_PLATEAU = [1.00, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30]

#: schedule name -> per-epoch level (``None`` = operator disabled)
LEVELS = {
    "none": None,
    # the campaign's reference schedule: annealed, then an exact bypass
    "plateau": [_PLATEAU[min(e // 3, 6)] if e < BYPASS_FROM else 0.0
                for e in range(EPOCHS)],
    # constant arms: never reach the target endpoint -> not continuations
    "const0.3": [0.30] * EPOCHS,
    "const0.5": [0.50] * EPOCHS,
    "const0.8": [0.80] * EPOCHS,
    "const1.0": [1.00] * EPOCHS,
}

RESOLUTIONS_ABL = {"R32": [32] * EPOCHS}

BUILDER = "continuation.ablation_ops.build_from_cell"

# rough per-run seconds on a T4, used only to balance the two job queues
_COST = {"none": 490, "plateau": 700, "const0.3": 760, "const0.5": 760,
         "const0.8": 760, "const1.0": 760}


def build_configs() -> list:
    """The 12 frozen configurations, in manifest order."""
    cfgs, seen = [], set()

    def add(cid, group, test, operator, levels, *, placement="conv_out",
            mask="all19", blurpool_sigma=None, constant=False,
            primary_path="target", rationale=""):
        if cid in seen:
            raise ValueError("duplicate configuration %s" % cid)
        seen.add(cid)
        cost = _COST[levels]
        if mask == "predown":
            cost = int(cost * 0.78)          # 2 filtered sites instead of 19
        if placement == "post_block":
            cost = int(cost * 0.88)          # 10 positions instead of 19
        if blurpool_sigma:
            cost += 25
        cfgs.append({"id": cid, "group": group, "test": test,
                     "operator": operator, "levels": levels,
                     "resolution": "R32", "reduction": "input_bilinear",
                     "placement": placement, "mask": mask,
                     "blurpool_sigma": blurpool_sigma, "constant": constant,
                     "primary_path": primary_path,
                     "controller_builder": BUILDER,
                     "diagnostics": True,
                     "rationale": rationale, "est_seconds": cost})

    # --- controls (fresh, mandatory) ---------------------------------------
    add("C_plain", "control", "control", "none", "none",
        rationale="no filter anywhere; the baseline every arm is measured against")
    add("C_plateau", "control", "control", "gaussian", "plateau",
        rationale="the campaign's reference arm, re-run on this batch's assets")

    # --- test 1: placement --------------------------------------------------
    add("P_postbn", "T1_placement", "1", "gaussian", "plateau", placement="post_bn",
        rationale="same 19 positions as the control but after BatchNorm; BN then "
                  "accumulates statistics on unfiltered activations")
    add("P_postblock", "T1_placement", "1", "gaussian", "plateau",
        placement="post_block",
        rationale="after the ReLU (10 positions): the only placement that is "
                  "anti-aliasing-correct at the two decimation points")

    # --- test 2: which sites ------------------------------------------------
    add("M_predown", "T2_mask", "2", "gaussian", "plateau", mask="predown",
        rationale="sites {6,12} only -- the conv outputs closest to a decimation. "
                  "Under H these carry most of the effect")
    add("M_nodown", "T2_mask", "2", "gaussian", "plateau", mask="nodown",
        rationale="the other 17 sites. Under H these carry almost nothing")

    # --- tests 3 and 4: constant sigma, no annealing ------------------------
    for lv, tag in (("const0.3", "030"), ("const0.5", "050"),
                    ("const0.8", "080"), ("const1.0", "100")):
        add("K_const%s" % tag, "T34_constant", "3+4", "gaussian", lv,
            constant=True, primary_path="current",
            rationale="fixed level, never annealed: architecture, not continuation. "
                      "Separates 'the filter helps' from 'annealing helps', and a "
                      "peak near sigma 0.5-0.8 would be a Nyquist signature")

    # --- test 5: a real, fixed anti-aliasing prefilter -----------------------
    add("B_blurpool", "T5_blurpool", "5", "none", "none", blurpool_sigma=0.5,
        rationale="fixed sigma 0.5 on the inputs of blocks[3] and blocks[6], "
                  "covering both strided convs and both decimating shortcuts; "
                  "no annealed Gaussian anywhere. The positive test of H")
    add("B_blurpool_plateau", "T5_blurpool", "5", "gaussian", "plateau",
        blurpool_sigma=0.5,
        rationale="BlurPool plus the annealed Gaussian: if they share a mechanism "
                  "their gains should be largely redundant")
    return cfgs


def build_cells() -> list:
    cells = []
    for c in build_configs():
        for s in SEEDS:
            cells.append({**c, "seed": s, "cell_id": "%s__seed%d" % (c["id"], s)})
    return cells


def assign(cells, n_jobs=2) -> list:
    """Longest-processing-time-first split into ``n_jobs`` balanced queues.

    Deterministic: the same cell list always yields the same assignment, so a job
    sees its own slice whichever machine runs it.  Each queue is returned already
    ordered longest-first, which is what the driver's shared worker queue wants.
    """
    queues = [[] for _ in range(n_jobs)]
    loads = [0] * n_jobs
    for c in sorted(cells, key=lambda c: (-c["est_seconds"], c["cell_id"])):
        j = loads.index(min(loads))
        queues[j].append(c)
        loads[j] += c["est_seconds"]
    return queues


def summary() -> dict:
    cells = build_cells()
    queues = assign(cells)
    return {"n_configs": len(build_configs()), "n_cells": len(cells),
            "seeds": list(SEEDS), "epochs": EPOCHS,
            "total_est_seconds": sum(c["est_seconds"] for c in cells),
            "jobs": [{"job": i, "n_cells": len(q),
                      "est_seconds": sum(c["est_seconds"] for c in q)}
                     for i, q in enumerate(queues)]}


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
    for c in build_configs():
        print("%-20s %-14s test=%-4s op=%-8s lv=%-9s place=%-11s mask=%-8s bp=%-4s primary=%s"
              % (c["id"], c["group"], c["test"], c["operator"], c["levels"],
                 c["placement"], c["mask"], c["blurpool_sigma"], c["primary_path"]))

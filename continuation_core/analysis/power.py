r"""Three guards that have to run before a number is read, and the correlation
protocol that survives them.

Each guard here caught a real error on the exploratory branch.  They are not
hygiene; they are the difference between a result and an artefact.

Guard 1 -- validity
-------------------
A probe rebuilds the model and loads a state dict.  An arm whose trained network
carries a *parameter-free* architectural change loads without error and is then
evaluated as a **different function**.  Metadata does not catch this, because it
tracks schedules and not architecture.  The check is therefore empirical: a cell
is admitted only if its probed test error reproduces the error recorded at
training time.  It caught two arms out of twenty-seven, one of them carrying the
most extreme value in the whole set.

Guard 2 -- reproducibility floor
--------------------------------
With no determinism flags, cuDNN's non-deterministic convolution backward places
two otherwise identical runs **6.4 to 7.5 apart** in weight space -- about half
the distance either travels from its initialisation -- while their test
accuracies agree to 0.12-0.28 pp.  Measured signatures moved only 1-2 % across
that, so they are properties of the recipe rather than of the draw; but
``lambda_max`` moved 8.2 %, and rankings closer than twice the floor are not
readable.  The floor has to be measured, not assumed, and every ranking is
reported against it.

Guard 3 -- detection ceiling
----------------------------
A correlation against a noisy target is attenuated.  With the training error
taken from a 500-image probe, the generalisation gap carried 1.3 points of
sampling noise against a real spread of 0.4, capping **any** correlation at
0.25 -- so a coefficient of 0.21 was read as a failure when it was at the
ceiling.  The ceiling is computed from the target's own noise and printed beside
every coefficient.

The protocol
------------
Four rules, each of which was learned by breaking:

* the gap comes from the **whole** pinned training subset, never a probe;
* correlations are **partial**, controlling the training error -- without that,
  a quantity that merely tracks how hard a network fitted its training set looks
  like a predictor (``||J||_F`` scored +0.81 raw and +0.22 partial);
* **Pearson and Spearman** are always reported together -- a single distant point
  manufactured coefficients of +0.94 that fell to +0.44 on ranks;
* correlations are computed **per condition**, not per checkpoint, or the sample
  size is a fiction: ten checkpoints that are four conditions are an n=4 test in
  an n=10 costume.
"""
from __future__ import annotations

import math


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy) if sx and sy else float("nan")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    return pearson(rank(xs), rank(ys))


def partial(xs, ys, zs):
    """``corr(x, y)`` with ``z`` held fixed."""
    rxy, rxz, ryz = pearson(xs, ys), pearson(xs, zs), pearson(ys, zs)
    den = math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / den if den > 0 else float("nan")


def p_two_sided(r, n):
    """Two-sided p for a correlation, via the Fisher z transform."""
    if n < 4 or not (-1 < r < 1):
        return float("nan")
    z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(n - 3)
    return math.erfc(abs(z) / math.sqrt(2))


def validity_gate(probed_error: dict, recorded_error: dict, tol: float = 1e-3) -> dict:
    """Admit only cells whose probed test error reproduces the recorded one."""
    rejected = {k: probed_error[k] - recorded_error[k] for k in probed_error
                if k in recorded_error and abs(probed_error[k] - recorded_error[k]) > tol}
    return {"tolerance": tol, "n_checked": len(probed_error),
            "rejected": rejected, "admitted": [k for k in probed_error if k not in rejected]}


def noise_floor(paired) -> dict:
    """Run-to-run standard deviation from paired repeats of identical cells.

    ``paired`` is an iterable of ``(value_run_a, value_run_b)``.  The per-run
    standard deviation is the paired-difference standard deviation over sqrt(2).
    """
    d = [float(b) - float(a) for a, b in paired]
    n = len(d)
    if n < 2:
        return {"n_pairs": n, "sd": None}
    m = sum(d) / n
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5
    return {"n_pairs": n, "sd": sd / math.sqrt(2), "mean_shift": m,
            "resolution": 2 * sd / math.sqrt(2)}


def detection_ceiling(observed_sd: float, measurement_sd: float) -> dict:
    """The largest correlation the target's own noise still permits."""
    true_var = max(observed_sd ** 2 - measurement_sd ** 2, 0.0)
    ceiling = math.sqrt(true_var) / observed_sd if observed_sd else 0.0
    return {"observed_sd": observed_sd, "measurement_sd": measurement_sd,
            "true_sd": math.sqrt(true_var), "ceiling": ceiling}


def correlate(rows, fields, *, target="test_error", control="train_error",
              n_tested: int | None = None) -> dict:
    """The full protocol, one row per **condition**.

    Every field is reported raw, on ranks, partialled on the control, with a
    two-sided p and -- when ``n_tested`` is given -- the Bonferroni threshold for
    the number of quantities actually tried, which is the honest denominator once
    a search has been run.
    """
    n = len(rows)
    y = [r[target] for r in rows]
    z = [r[control] for r in rows]
    alpha = 0.05 / n_tested if n_tested else 0.05
    out = []
    for f in fields:
        if any(r.get(f) is None for r in rows):
            out.append({"field": f, "skipped": "missing in at least one condition"})
            continue
        x = [r[f] for r in rows]
        pr = partial(x, y, z)
        p = p_two_sided(pr, n - 1)
        out.append({"field": f, "pearson": pearson(x, y), "spearman": spearman(x, y),
                    "partial": pr, "p_partial": p,
                    "passes_corrected": bool(p == p and p < alpha),
                    "corr_with_control": pearson(x, z)})
    return {"n_conditions": n, "target": target, "control": control,
            "n_quantities_tested": n_tested, "alpha_corrected": alpha,
            "rows": out,
            "note": "correlations are per condition; per-checkpoint figures inflate n"}

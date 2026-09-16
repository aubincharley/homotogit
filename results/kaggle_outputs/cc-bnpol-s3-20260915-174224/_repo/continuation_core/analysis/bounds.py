r"""The PAC-Bayes bound, computed so that its failure is quantified rather than asserted.

What this is for
----------------
The standard reading of a flatness result is: the solutions tolerate a wider
posterior, a PAC-Bayes bound pays for exactly that, so the bound should explain
the gain.  On the exploratory branch it did not, and by a margin worth recording
precisely: the bound was vacuous at 0.993-0.995, ``KL / n`` sitting near 200
where it needs to be below about 0.5.

Three things were tried against that factor and each is measured here rather
than guessed:

* the **flatness itself** -- a 1.285x wider posterior divides the KL by 1.65;
* the **BatchNorm gauge** -- see :mod:`.gauge`; it removed 3 % of the distance,
  so 1.06 on the KL;
* leaving the bound's **form** alone, since the kl-form of Maurer inverted
  numerically is already the tight one.

1.65 and 1.06 against a needed 400.  The conclusion is not that the bound is
loose but that the distance ``||W* - W0|| ~ 13.6`` is *functional*: the network
really travels that far, and no reparametrisation argument recovers it.

Conventions
-----------
Prior centred at the **pinned initialisation**, which is legitimate only because
it was fixed before training and kept.  Posterior ``N(W*, tau^2 I)``, so

    KL(Q||P) = ||W* - W0||^2 / (2 tau^2)

and the bound is the kl-form inverted for the empirical 0-1 error.  The 0-1
error, not the cross-entropy, is what that form needs, since it requires a loss
in ``[0,1]``.  A union bound over the ``tau`` grid is applied, because ``tau`` is
chosen after seeing the curve.

A filter-normalised ``tau`` is **not** admissible here: it is not the standard
deviation of an isotropic posterior, and only the isotropic one admits this
closed-form KL.  The function refuses it rather than returning a number that
looks comparable.
"""
from __future__ import annotations

import math


def kl_bernoulli(q: float, p: float) -> float:
    """``kl(q || p)`` for Bernoulli parameters, in nats."""
    def term(a, b):
        if a == 0:
            return 0.0
        if b <= 0:
            return float("inf")
        return a * math.log(a / b)
    return term(q, p) + term(1 - q, 1 - p)


def kl_inverse(q: float, c: float, tol: float = 1e-10) -> float:
    """Largest ``p >= q`` with ``kl(q || p) <= c``, by bisection."""
    if c <= 0:
        return q
    lo, hi = q, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(q, mid) <= c:
            lo = mid
        else:
            hi = mid
    return lo


def kl_from_distance(distance: float, tau: float) -> float:
    """``||W* - W0||^2 / (2 tau^2)`` -- the isotropic Gaussian KL."""
    if tau <= 0:
        return float("inf")
    return distance ** 2 / (2.0 * tau ** 2)


def bound(empirical_error: float, distance: float, tau: float, n: int,
          delta: float = 0.05, n_tau: int = 1, mode: str = "absolute") -> dict:
    """Maurer's kl-form bound at one posterior width.

    ``n_tau`` applies the union bound over a grid of widths, since the width is
    chosen after seeing the risk curve.
    """
    if mode != "absolute":
        raise ValueError("only an isotropic (absolute) tau admits this closed-form KL; "
                         "a filter-normalised tau is a relative perturbation and is "
                         "not the standard deviation of an isotropic posterior")
    kl = kl_from_distance(distance, tau)
    c = (kl + math.log(2.0 * math.sqrt(n) * n_tau / delta)) / n
    return {"tau": tau, "kl": kl, "kl_over_n": kl / n, "complexity_term": c,
            "empirical_error": empirical_error,
            "bound": kl_inverse(empirical_error, c),
            "n": n, "delta": delta, "n_tau_union": n_tau,
            "vacuous": bool(kl_inverse(empirical_error, c) >= 1.0 - 1e-9)}


def best_bound(curve, distance: float, n: int, delta: float = 0.05) -> dict:
    """The tightest bound over a risk curve, and how far it is from useful.

    ``curve`` is an iterable of ``(tau, empirical_error_at_tau)``.  The report
    carries the factor by which ``KL / n`` would have to fall, which is the
    honest way to say how far the route is from working -- rather than reporting
    a number near 1 and calling it a bound.
    """
    rows = [bound(err, distance, tau, n, delta, n_tau=len(list(curve)) or 1)
            for tau, err in curve if tau > 0]
    if not rows:
        return {"rows": [], "best": None}
    best = min(rows, key=lambda r: r["bound"])
    return {"rows": rows, "best": best,
            "all_vacuous": all(r["vacuous"] for r in rows),
            "kl_over_n_at_best": best["kl_over_n"],
            "factor_needed": best["kl_over_n"] / 0.5 if best["kl_over_n"] > 0 else None,
            "note": ("factor_needed is how much smaller KL/n must be for the bound "
                     "to be informative; flatness bought 1.65 and the BatchNorm "
                     "gauge 1.06 on the exploratory branch")}

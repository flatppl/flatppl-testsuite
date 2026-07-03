#!/usr/bin/env python3
"""INDEPENDENT closed-form oracle for the sample-path numeric gate's
hierarchical-Normal model (``corpora/sample/hier_normal/hier_normal.flatppl``):

    mu       ~ Normal(0, 10)
    y1 | mu  ~ Normal(mu, 1)
    y2 | mu  ~ Normal(mu, 1)

y1 and y2 share the SAME draw of mu (not two independent draws) — that
shared ancestry is exactly what ``suites/sample_gate.py``'s
``cov(y1, y2)`` check exists to catch in the determinized FlatPDL.

Used by ``gen_expected.py`` (structural moments -> ``expected.json``) and by
``suites/sample_gate.py``'s density-consistency check (per-point joint
log-density). Never derived from flatppl-js output — this is scipy's
closed-form Normal, independent of any FlatPPL engine.
"""
from __future__ import annotations

from scipy.stats import norm

MU_SIGMA = 10.0
Y_SIGMA = 1.0


def moments() -> dict[str, float]:
    """Structural (N-independent) closed-form moments of the joint law.

    mu ~ N(0, 10) => E[mu] = 0, Var[mu] = 100.
    y_i = mu + eps_i, eps_i ~ N(0, 1) independent of mu and of each other
    => E[y_i] = 0, Var[y_i] = Var[mu] + 1 = 101.
    Cov(y1, y2) = Var[mu] = 100, because y1 and y2 share the same mu term
    and their independent noise terms eps_1, eps_2 don't covary.
    """
    var_mu = MU_SIGMA ** 2
    var_y = var_mu + Y_SIGMA ** 2
    cov_y1_y2 = var_mu
    return {
        "mean_mu": 0.0, "var_mu": var_mu,
        "mean_y1": 0.0, "var_y1": var_y,
        "mean_y2": 0.0, "var_y2": var_y,
        "cov_y1_y2": cov_y1_y2,
    }


def logdensity(mu: float, y1: float, y2: float) -> float:
    """Closed-form joint log-density at one (mu, y1, y2) point:
    log N(mu; 0, 10) + log N(y1; mu, 1) + log N(y2; mu, 1)."""
    return (
        norm.logpdf(mu, 0.0, MU_SIGMA)
        + norm.logpdf(y1, mu, Y_SIGMA)
        + norm.logpdf(y2, mu, Y_SIGMA)
    )

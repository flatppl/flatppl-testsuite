"""INDEPENDENT closed-form oracle for the hierarchical-Normal sample-path
model (``hier_normal.flatppl`` / ``hier_normal_density.flatppl``):

    mu       ~ Normal(0, 10)
    y1 | mu  ~ Normal(mu, 1)
    y2 | mu  ~ Normal(mu, 1)

y1 and y2 share the SAME draw of mu (not two independent draws) -- that
shared ancestry is exactly what this directory's ``cov_y1_y2`` check
(``test.json``) exists to catch in the determinized FlatPDL.

Lifted verbatim (moments/logdensity formulas unchanged) from the legacy
``corpora/sample/oracle.py``, which this directory's ``test.json`` `checks`
`expected`/`atol` values were originally frozen from via
``corpora/sample/gen_expected.py`` (N=4000, seed_base=0, tolerance = 5 *
Monte-Carlo SE -- see that script for the derivation, mirrored below in
``stat()``). Never derived from flatppl-js output -- this is scipy's
closed-form Normal, independent of any FlatPPL engine.

``stat()`` runs offline under ``unified/regen.py`` to refreeze the
``sample_stats`` checks' `expected`/`atol`. ``logdensity()`` is also called
live, at test time, by the unified ``(sample, det-js)`` runner
(``unified/runners/sample_detjs.py``) for the ``density_consistency``
check -- that check has no frozen scalar to compare against because the
points it scores are only known once the seed-sweep has run, so the runner
loads this module and evaluates the oracle per swept point.
"""
from __future__ import annotations

import math

from scipy.stats import norm

MU_SIGMA = 10.0
Y_SIGMA = 1.0
N_SAMPLES = 4000
K = 5  # tolerance = K * Monte-Carlo SE (see corpora/sample/gen_expected.py)


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


def stat() -> dict[str, dict[str, float]]:
    """Refreeze the `sample_stats` checks' `expected`/`atol` for
    `unified/regen.py`: `expected` is the structural closed-form moment
    (N-independent); `atol` is `K` times the Monte-Carlo standard error for
    this directory's frozen `n_samples` (see `corpora/sample/gen_expected.py`
    for the SE derivations -- reproduced verbatim here). Returns
    `{check_id: {"expected": ..., "atol": ...}}`, one entry per
    `sample_stats` check id in this directory's `test.json`."""
    m = moments()
    var_mu = m["var_mu"]
    var_y1, var_y2 = m["var_y1"], m["var_y2"]
    assert var_y1 == var_y2
    var_y = var_y1
    cov_y1_y2 = m["cov_y1_y2"]

    se_mean_mu = math.sqrt(var_mu / N_SAMPLES)
    se_mean_y = math.sqrt(var_y / N_SAMPLES)
    se_var_mu = var_mu * math.sqrt(2 / N_SAMPLES)
    se_var_y = var_y * math.sqrt(2 / N_SAMPLES)
    se_cov = math.sqrt((var_y1 * var_y2 + cov_y1_y2 ** 2) / N_SAMPLES)

    return {
        "mean_mu": {"expected": m["mean_mu"], "atol": K * se_mean_mu},
        "mean_y1": {"expected": m["mean_y1"], "atol": K * se_mean_y},
        "mean_y2": {"expected": m["mean_y2"], "atol": K * se_mean_y},
        "var_mu": {"expected": var_mu, "atol": K * se_var_mu},
        "var_y1": {"expected": var_y1, "atol": K * se_var_y},
        "var_y2": {"expected": var_y2, "atol": K * se_var_y},
        "cov_y1_y2": {"expected": cov_y1_y2, "atol": K * se_cov},
    }


def logdensity(mu: float, y1: float, y2: float) -> float:
    """Closed-form joint log-density at one (mu, y1, y2) point:
    log N(mu; 0, 10) + log N(y1; mu, 1) + log N(y2; mu, 1)."""
    return (
        norm.logpdf(mu, 0.0, MU_SIGMA)
        + norm.logpdf(y1, mu, Y_SIGMA)
        + norm.logpdf(y2, mu, Y_SIGMA)
    )

"""INDEPENDENT closed-form oracle for the bivariate `ksuperpose` mixture
sample path (`model.flatppl` / `model_density.flatppl`):

    mix = normalize(ksuperpose(MvNormal, [0.3, 0.7])(mu = mus, cov = covs))
    mus = [[-2, 1], [3, -1]]
    covs = [[[1, 0.3], [0.3, 1]], [[0.5, 0], [0, 2]]]

Mixture moments, exact:

    mean = sum_i w_i mu_i                               = [1.5, -0.4]
    cov  = sum_i w_i (cov_i + mu_i mu_i^T) - mean mean^T = [[5.9, -2.01],
                                                            [-2.01, 2.54]]

Tolerances are 5 Monte-Carlo standard errors at N = 4000, following
`corpora/sample/gen_expected.py`'s rule. The mixture is bimodal, so the
variance and covariance SEs use the mixture's own fourth moments rather
than a Gaussian approximation:

    SE(mean_j) = sqrt(m2_j / N)
    SE(var_j)  = sqrt((m4_j - m2_j^2) / N)
    SE(cov)    = sqrt((E[u^2 v^2] - cov^2) / N),  u, v centred

giving 5 SE = 0.192029 / 0.125996 (means), 0.512058 / 0.257920
(variances), 0.256891 (covariance). Never derived from engine output.

STATUS: the rust determiniser REFUSES the sample path — "ksuperpose ... not
implemented" for a sampled mixture (the component-index draw is not built),
which is not specific to the multivariate family; the density path refuses
separately on the multivariate family itself (see `mv_mixture`). Observed
live 2026-09-01. `allow_skip: true`.

The flatppl-js engine samples this mixture correctly today, verified
against these moments in
`flatppl-js/packages/engine/test/ksuperpose-multivariate.test.ts`.
"""
from __future__ import annotations

import math

import numpy as np

W = np.array([0.3, 0.7])
MUS = np.array([[-2.0, 1.0], [3.0, -1.0]])
COVS = [
    np.array([[1.0, 0.3], [0.3, 1.0]]),
    np.array([[0.5, 0.0], [0.0, 2.0]]),
]
N_SAMPLES = 4000
K = 5  # tolerance = K * Monte-Carlo SE


def _mean() -> np.ndarray:
    return sum(w * m for w, m in zip(W, MUS))


def _cov() -> np.ndarray:
    mean = _mean()
    second = sum(w * (c + np.outer(m, m)) for w, m, c in zip(W, MUS, COVS))
    return second - np.outer(mean, mean)


def _central(k: int, j: int) -> float:
    """E[(x_j - mean_j)^k] for the mixture, k in {2, 4}."""
    mean = _mean()
    total = 0.0
    for w, m, c in zip(W, MUS, COVS):
        d = m[j] - mean[j]
        s2 = c[j, j]
        if k == 2:
            total += w * (s2 + d * d)
        elif k == 4:
            total += w * (3 * s2 * s2 + 6 * s2 * d * d + d ** 4)
        else:
            raise ValueError(f"unsupported moment {k}")
    return float(total)


def moments() -> dict[str, float]:
    mean, cov = _mean(), _cov()
    return {
        "mean_y1": float(mean[0]),
        "mean_y2": float(mean[1]),
        "var_y1": float(cov[0, 0]),
        "var_y2": float(cov[1, 1]),
        "cov_y1_y2": float(cov[0, 1]),
    }


def stat(name: str) -> tuple[float, float]:
    """(expected, atol) for one `sample_stats` check, refrozen offline by regen."""
    mean, cov = _mean(), _cov()
    n = N_SAMPLES
    if name in ("mean_y1", "mean_y2"):
        j = 0 if name.endswith("y1") else 1
        return float(mean[j]), K * math.sqrt(_central(2, j) / n)
    if name in ("var_y1", "var_y2"):
        j = 0 if name.endswith("y1") else 1
        m2, m4 = _central(2, j), _central(4, j)
        return float(cov[j, j]), K * math.sqrt((m4 - m2 * m2) / n)
    if name == "cov_y1_y2":
        mean_v = mean
        fourth = 0.0
        for w, m, c in zip(W, MUS, COVS):
            d0, d1 = m[0] - mean_v[0], m[1] - mean_v[1]
            fourth += w * (
                c[0, 0] * c[1, 1] + 2 * c[0, 1] ** 2
                + c[0, 0] * d1 * d1 + c[1, 1] * d0 * d0
                + 4 * c[0, 1] * d0 * d1 + d0 * d0 * d1 * d1
            )
        return float(cov[0, 1]), K * math.sqrt((fourth - cov[0, 1] ** 2) / n)
    raise KeyError(name)


def logdensity(y1: float, y2: float) -> float:
    """The mixture's joint log-density at one swept realization, for the
    `density_consistency` check."""
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal

    y = np.array([y1, y2], dtype=float)
    terms = [
        math.log(w) + multivariate_normal(mean=m, cov=c).logpdf(y)
        for w, m, c in zip(W, MUS, COVS)
    ]
    return float(logsumexp(terms))

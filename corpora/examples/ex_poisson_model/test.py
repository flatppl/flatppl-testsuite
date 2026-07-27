"""Independent oracle for ex_poisson_model.

Conjugate Poisson-Gamma: ``lambda ~ Gamma(shape=2, rate=1)``; ``y ~
iid(Poisson(lambda), 5)`` against the fixed ``counts_data``.
"""
from scipy.stats import gamma as gamma_dist, poisson

_COUNTS = [2, 3, 7, 6, 4]


def oracle(point: dict) -> float:
    lam = point["lambda"]
    lp = gamma_dist.logpdf(lam, a=2.0, scale=1.0)
    lp += sum(poisson.logpmf(c, lam) for c in _COUNTS)
    return float(lp)

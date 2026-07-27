"""Independent oracle for ex_poisson_glm_link.

``intercept, slope ~ Normal(0,1)``; ``eta_i = intercept + slope*x_i``, ``mu_i
= exp(eta_i)``, ``y_i ~ Poisson(mu_i)`` against the fixed placeholder
covariate/count data.
"""
import math

from scipy.stats import norm, poisson

_X = [-1.0, 0.2, 0.5, 1.3, 2.1]
_Y = [0, 1, 2, 3, 8]


def oracle(point: dict) -> float:
    intercept, slope = point["intercept"], point["slope"]
    lp = norm.logpdf(intercept, 0.0, 1.0) + norm.logpdf(slope, 0.0, 1.0)
    for xi, yi in zip(_X, _Y):
        eta = intercept + slope * xi
        mu = math.exp(eta)
        lp += poisson.logpmf(yi, mu)
    return float(lp)

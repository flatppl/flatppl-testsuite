"""Independent oracle for ex_plant_growth_group_effects.

Log-posterior ``logdensityof(Pi_post, point)``: the log-likelihood below plus
the log-prior ``Normal(5, 2)`` on ``mu`` and ``Exponential(1)`` on both scales.
The zero-sum effects use the normalized Gaussian law in its two free coordinates,
with covariance ``effect_scale**2 * (I - ones(2, 2) / 3)``.

One-way group effects on R's ``PlantGrowth`` weights: ``weight_i ~
Normal(mu + effects[g_i], sigma)`` with ten plants in each of three groups.
"""
import numpy as np
from scipy.stats import expon, multivariate_normal, norm

_WEIGHTS = [
    4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14,
    4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69,
    6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26,
]


def oracle(point: dict) -> float:
    mu, effects, sigma = point["mu"], point["effects"], point["sigma"]
    lp = sum(norm.logpdf(w, mu + effects[i // 10], sigma) for i, w in enumerate(_WEIGHTS))
    lp += norm.logpdf(mu, 5.0, 2.0) + expon.logpdf(sigma, scale=1.0)
    scale = point["effect_scale"]
    covariance = scale**2 * (np.eye(2) - np.ones((2, 2)) / 3)
    lp += multivariate_normal.logpdf(effects[:2], mean=np.zeros(2), cov=covariance)
    lp += expon.logpdf(scale, scale=1.0)
    return float(lp)

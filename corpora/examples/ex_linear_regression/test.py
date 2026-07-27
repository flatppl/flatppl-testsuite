"""Independent oracle for ex_linear_regression.

``sigma2 ~ InverseGamma(5,5)``, ``sigma = sqrt(sigma2)`` (deterministic-
reparam draw field, same sqrt-pushforward Jacobian as ``dissimilar_mixture``);
``alpha, beta ~ Normal(0, sigma*3)`` -- a DEPENDENT prior (each conditioned on
the drawn ``sigma``), so the joint prior log-density is the sqrt-pushforward
term for ``sigma`` plus the two conditional Normal log-densities evaluated at
that same ``sigma``; ``means = alpha + beta * x_data``, ``y ~
Normal.(means, sigma)`` against the fixed ``x_data``/``y_data``.
"""
import math

from scipy.stats import invgamma, norm

_X = [1.1, 1.5, 1.3, 1.4]
_Y = [3.2, 4.1, 3.4, 3.9]


def oracle(point: dict) -> float:
    alpha, beta, sigma = point["alpha"], point["beta"], point["sigma"]
    lp = invgamma.logpdf(sigma**2, a=5.0, scale=5.0) + math.log(2.0) + math.log(sigma)
    lp += norm.logpdf(alpha, 0.0, sigma * 3.0)
    lp += norm.logpdf(beta, 0.0, sigma * 3.0)
    means = [alpha + beta * x for x in _X]
    lp += sum(norm.logpdf(y, m, sigma) for y, m in zip(_Y, means))
    return float(lp)

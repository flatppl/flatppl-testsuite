"""Independent oracle for ex_dissimilar_mixture.

``p ~ Beta(2,2)``; ``mu ~ Normal(0,1)``, ``sigma2 ~ InverseGamma(2,2)``,
``sigma = sqrt(sigma2)`` (deterministic-reparam draw field: per the
``pushfwd`` engine contract, ``sqrt = pow(_, 1/2)`` has inverse ``y -> y**2``
and forward log-volume ``log(1/2) - log(x)``, giving ``logdensityof(sigma) =
logdensityof_M(sigma**2) + log(2*sigma)``); ``normal_mixand = Normal(mu,
sigma)``; ``shape, rate ~`` half-Normal(0,5) (independent); ``gamma_mixand =
Gamma(shape, rate)``; ``mixture = normalize(superpose(weighted(p,
normal_mixand), weighted(1-p, gamma_mixand)))`` -- a no-op ``normalize``
since the weights sum to 1 and each mixand already has total mass 1; ``y ~
iid(mixture, 20)`` against the fixed ``y_obs`` (includes negative values,
only reachable by ``normal_mixand``).
"""
import math

from scipy.special import logsumexp
from scipy.stats import beta as beta_dist, gamma as gamma_dist, invgamma, norm

_Y = [
    7.23, 5.13, 1.20, -0.33, 0.23,
    -0.22, 1.34, 0.80, 0.50, -0.75,
    3.79, 0.01, -0.76, 0.21, 1.48,
    1.21, 12.11, 15.96, 9.83, 3.92,
]


def oracle(point: dict) -> float:
    p, mu, sigma = point["p"], point["mu"], point["sigma"]
    shape, rate = point["shape"], point["rate"]
    lp = beta_dist.logpdf(p, 2.0, 2.0)
    lp += norm.logpdf(mu, 0.0, 1.0)
    lp += invgamma.logpdf(sigma**2, a=2.0, scale=2.0) + math.log(2.0) + math.log(sigma)
    lp += math.log(2.0) + norm.logpdf(shape, 0.0, 5.0)
    lp += math.log(2.0) + norm.logpdf(rate, 0.0, 5.0)
    for y in _Y:
        log_normal = math.log(p) + norm.logpdf(y, mu, sigma)
        log_gamma = math.log(1.0 - p) + gamma_dist.logpdf(y, a=shape, scale=1.0 / rate)
        lp += logsumexp([log_normal, log_gamma])
    return float(lp)

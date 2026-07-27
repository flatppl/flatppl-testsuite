"""Independent oracle for ex_gamma_reparam.

``mu ~ Normal(0, 5)``; ``sigma ~ Gamma(gamma_shape_rate(2.0, 1.0))`` -- the
reparameterization is applied to LITERAL constants (mean=2.0, sd=1.0), so it
resolves to the static prior ``Gamma(shape=4, rate=2)``, not a function of the
draws; ``y ~ iid(Normal(mu, sigma), 5)`` against the fixed ``y_data``.
"""
from scipy.stats import gamma as gamma_dist, norm

_Y = [1.8, 2.3, 1.1, 2.9, 2.0]
_SHAPE, _RATE = 2.0**2 / 1.0**2, 2.0 / 1.0**2  # gamma_shape_rate(2.0, 1.0)


def oracle(point: dict) -> float:
    mu, sigma = point["mu"], point["sigma"]
    lp = norm.logpdf(mu, 0.0, 5.0) + gamma_dist.logpdf(sigma, a=_SHAPE, scale=1.0 / _RATE)
    lp += sum(norm.logpdf(y, mu, sigma) for y in _Y)
    return float(lp)

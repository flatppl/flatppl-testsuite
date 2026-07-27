"""Independent oracle for ex_partial_pooling.

``phi ~ Uniform(interval(0,1))``, ``kappa ~ Gamma(shape=2, rate=0.05)``;
``alpha = phi*kappa``, ``beta = (1-phi)*kappa``; ``theta ~ iid(Beta(alpha,
beta), 8)``; ``y ~ Binomial.(at_bats_data, theta)`` against the fixed
placeholder hits/at-bats data.
"""
from scipy.stats import beta as beta_dist, binom, gamma as gamma_dist, uniform

_AT_BATS = [45, 45, 45, 60, 20, 100, 15, 250]
_HITS = [12, 15, 10, 18, 8, 29, 6, 71]


def oracle(point: dict) -> float:
    phi, kappa = point["phi"], point["kappa"]
    theta = point["theta"]
    alpha = phi * kappa
    beta_param = (1.0 - phi) * kappa
    lp = uniform.logpdf(phi, loc=0.0, scale=1.0) + gamma_dist.logpdf(
        kappa, a=2.0, scale=1.0 / 0.05
    )
    lp += sum(beta_dist.logpdf(ti, alpha, beta_param) for ti in theta)
    lp += sum(binom.logpmf(h, n, ti) for h, n, ti in zip(_HITS, _AT_BATS, theta))
    return float(lp)

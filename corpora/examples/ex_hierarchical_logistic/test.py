"""Independent oracle for ex_hierarchical_logistic.

``mu_a ~ Normal(0,1)``, ``sigma_a ~ Gamma(shape=4, rate=2)``, ``a ~
iid(Normal(mu_a, sigma_a), 3)``, ``b ~ locscale(StudentT(3), 0, 2.5)``;
``eta_i = a[group_i] + b*x_i``, ``y_i ~ Bernoulli(invlogit(eta_i))`` against
the fixed placeholder data (``group_data`` is 1-based).
"""
from scipy.stats import bernoulli, gamma as gamma_dist, norm, t as t_dist

_X = [-1.2, 0.4, 1.1, -0.3, 0.8, 2.0]
_GROUP = [1, 2, 3, 1, 2, 3]
_Y = [0, 1, 1, 0, 1, 1]


def _invlogit(z: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-z))


def oracle(point: dict) -> float:
    mu_a, sigma_a, b = point["mu_a"], point["sigma_a"], point["b"]
    a = point["a"]
    lp = (
        norm.logpdf(mu_a, 0.0, 1.0)
        + gamma_dist.logpdf(sigma_a, a=4.0, scale=1.0 / 2.0)
        + sum(norm.logpdf(ag, mu_a, sigma_a) for ag in a)
        + t_dist.logpdf(b, 3.0, 0.0, 2.5)
    )
    for xi, gi, yi in zip(_X, _GROUP, _Y):
        eta = a[gi - 1] + b * xi
        lp += bernoulli.logpmf(yi, _invlogit(eta))
    return float(lp)

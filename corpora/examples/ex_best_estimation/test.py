"""Independent oracle for ex_best_estimation.

BEST (Kruschke 2013): ``mu1, mu2 ~ Normal(100, 20)``, ``sigma1, sigma2 ~
Uniform(interval(0.1, 20))``, ``nu ~ Exponential(rate=1/29)``; ``y1 ~
iid(locscale(StudentT(nu), mu1, sigma1), 8)``, ``y2`` likewise with group 2's
params, against the fixed placeholder data.
"""
from scipy.stats import expon, norm, t as t_dist, uniform

_Y1 = [101.0, 100.0, 102.0, 104.0, 100.0, 103.0, 99.0, 105.0]
_Y2 = [99.0, 101.0, 100.0, 98.0, 100.0, 97.0, 102.0, 100.0]


def oracle(point: dict) -> float:
    mu1, mu2 = point["mu1"], point["mu2"]
    sigma1, sigma2 = point["sigma1"], point["sigma2"]
    nu = point["nu"]
    lp = (
        norm.logpdf(mu1, 100.0, 20.0)
        + norm.logpdf(mu2, 100.0, 20.0)
        + uniform.logpdf(sigma1, loc=0.1, scale=19.9)
        + uniform.logpdf(sigma2, loc=0.1, scale=19.9)
        + expon.logpdf(nu, scale=29.0)
    )
    lp += sum(t_dist.logpdf(y, nu, mu1, sigma1) for y in _Y1)
    lp += sum(t_dist.logpdf(y, nu, mu2, sigma2) for y in _Y2)
    return float(lp)

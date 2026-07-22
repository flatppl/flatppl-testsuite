"""Independent scipy oracle: MvNormal(mu, cov) logdensity at the fixed
observed a=[0.2, 0.1]."""
from scipy.stats import multivariate_normal


def oracle(point: dict) -> float:
    mu = point["mu"]
    cov = point["cov"]
    return float(multivariate_normal.logpdf([0.2, 0.1], mean=mu, cov=cov))

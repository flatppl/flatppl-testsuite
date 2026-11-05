"""Independent oracle for frag_exponential_normal_marginal_tail.

`v ~ Exponential(rate=1.0)`, `y | v ~ Normal(mu=v, sigma=1.0)`; the marginal of y is
the exponentially modified Gaussian, scored at y=-38.0, deep in the left tail, where erfc(z) underflows.
"""
from scipy.stats import exponnorm


def oracle() -> float:
    return float(exponnorm.logpdf(-38.0, 1.0 / (1.0 * 1.0), loc=0.0, scale=1.0))

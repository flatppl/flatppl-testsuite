"""Independent oracle for frag_exponential_normal_marginal_narrow.

`v ~ Exponential(rate=30.0)`, `y | v ~ Normal(mu=v, sigma=1.0)`; the marginal of y is
the exponentially modified Gaussian, scored at y=0.5, with a narrow Normal, where float32 erfc(z) underflows at the mode.
"""
from scipy.stats import exponnorm


def oracle() -> float:
    return float(exponnorm.logpdf(0.5, 1.0 / (1.0 * 30.0), loc=0.0, scale=1.0))

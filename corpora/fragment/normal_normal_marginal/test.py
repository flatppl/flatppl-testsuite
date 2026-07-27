"""Independent oracle for frag_normal_normal_marginal.

`z ~ Normal(mu=0.0, sigma=1.0)`, `y | z ~ Normal(mu=z, sigma=2.0)`; the
Normal-Normal conjugate marginal of y is `Normal(mu=0.0, sigma=sqrt(1.0^2 +
2.0^2))`, scored at y=1.5.
"""
import math

from scipy.stats import norm


def oracle() -> float:
    return norm.logpdf(1.5, loc=0.0, scale=math.sqrt(1.0**2 + 2.0**2))

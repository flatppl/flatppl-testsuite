"""Independent oracle for frag_special_functions_polygamma_mean.

`x ~ Normal(mu = digamma(1.7) + polygamma(1, 1.7), sigma = 1.0)` (spec §09
special-functions), scored at x = 0.5.
"""
from scipy import special
from scipy.stats import norm


def oracle() -> float:
    return float(norm.logpdf(0.5, loc=special.digamma(1.7) + special.polygamma(1, 1.7), scale=1.0))

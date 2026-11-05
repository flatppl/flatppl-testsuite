"""Independent oracle for frag_special_functions_erf_mean.

`x ~ Normal(mu = erf(0.3), sigma = 1.0)` (spec §09 special-functions), scored at
x = 0.5.
"""
from scipy.special import erf
from scipy.stats import norm


def oracle() -> float:
    return float(norm.logpdf(0.5, loc=erf(0.3), scale=1.0))

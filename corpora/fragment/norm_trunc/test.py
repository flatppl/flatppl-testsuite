"""Independent oracle for frag_norm_trunc.

`normalize(truncate(...))` renormalizes by the interval's probability mass;
this is exactly `scipy.stats.truncnorm.logpdf`.
"""
import math

from scipy.stats import norm


def oracle() -> float:
    return norm.logpdf(0.5, 0, 1) - math.log(norm.cdf(2, 0, 1) - norm.cdf(-1, 0, 1))

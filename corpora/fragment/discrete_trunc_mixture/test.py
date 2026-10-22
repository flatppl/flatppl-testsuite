"""Independent oracle for frag_discrete_trunc_mixture.

`truncate` keeps the atoms its closed interval holds (§06, §03), so the
normalizer is the base's pmf summed over them; this scores a mixture of two truncated discrete laws, both at their closed lower end 1.
"""
import math

from scipy.stats import poisson


def oracle() -> float:
    P = poisson(3.0)
    density = 0.3 * P.pmf(1) + 0.7 * 0.2
    mass = 0.3 * (P.cdf(4) - P.cdf(0)) + 0.7 * (0.2 + 0.3)
    return math.log(density) - math.log(mass)

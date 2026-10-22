"""Independent oracle for frag_poisson_trunc_zero.

`truncate` keeps the atoms its closed interval holds (§06, §03), so the
normalizer is the base's pmf summed over them; this scores the atom at the closed lower end 0.
"""
import math

from scipy.stats import poisson


def oracle() -> float:
    P = poisson(3.0)
    return P.logpmf(0) - math.log(P.cdf(5))

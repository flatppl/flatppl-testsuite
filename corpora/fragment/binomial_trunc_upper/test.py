"""Independent oracle for frag_binomial_trunc_upper.

`truncate` keeps the atoms its closed interval holds (§06, §03), so the
normalizer is the base's pmf summed over them; this scores the atom at the closed upper end 10.
"""
import math

from scipy.stats import binom


def oracle() -> float:
    B = binom(10, 0.3)
    return B.logpmf(10) - math.log(B.sf(2))

"""Independent oracle for frag_poisson_trunc_tail.

`truncate` keeps the atoms its closed interval holds (§06, §03), so the
normalizer is the base's pmf summed over them; this scores the atom at the closed lower end 2 of an unbounded interval.
"""
import math

from scipy.stats import poisson


def oracle() -> float:
    P = poisson(3.0)
    return P.logpmf(2) - P.logsf(1)

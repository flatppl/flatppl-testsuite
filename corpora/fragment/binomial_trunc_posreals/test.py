"""Independent oracle for frag_binomial_trunc_posreals.

On integer atoms a §03 named set keeps an atom range, and the normalizer is the
base's pmf summed over it; this scores the atom 1, the lowest a discrete law
keeps in `posreals` (0, inf].
"""
import math

from scipy.stats import binom


def oracle() -> float:
    B = binom(10, 0.3)
    return B.logpmf(1) - B.logsf(0)

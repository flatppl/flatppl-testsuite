"""Independent oracle for frag_poisson_trunc_posintegers.

On integer atoms a §03 named set keeps an atom range, and the normalizer is the
base's pmf summed over it; this scores the lowest kept atom 1 of `posintegers`.
"""
import math

from scipy.stats import poisson


def oracle() -> float:
    P = poisson(3.0)
    return P.logpmf(1) - P.logsf(0)

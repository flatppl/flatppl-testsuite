"""Independent oracle for frag_named_set_trunc_mixture.

On integer atoms a §03 named set keeps an atom range, and the normalizer is the
base's pmf summed over it; this scores the atom 0, which `posreals` drops from
the Poisson component and `nonnegintegers` keeps in the Geometric one.
"""
import math

from scipy.stats import geom, poisson


def oracle() -> float:
    P = poisson(3.0)
    G = geom(0.4, loc=-1)  # failures before the first success
    return math.log(0.7 * G.pmf(0)) - math.log(0.3 * P.sf(0) + 0.7 * G.sf(-1))

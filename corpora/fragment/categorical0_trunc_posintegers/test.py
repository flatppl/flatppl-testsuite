"""Independent oracle for frag_categorical0_trunc_posintegers.

On integer atoms a §03 named set keeps an atom range, and the normalizer is the
base's pmf summed over it; this scores the atom 1 of a Categorical0 that
`posintegers` keeps with 2.
"""
import math


def oracle() -> float:
    return math.log(0.3 / (0.3 + 0.5))

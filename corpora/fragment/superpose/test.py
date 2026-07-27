"""Independent oracle for frag_superpose.

`m = superpose(Normal(0,1), Normal(1,2))`; superpose is measure addition
(§06) -- the two component DENSITIES (not a normalized mixture) add at a
point.
"""
import math

from scipy.stats import norm


def oracle() -> float:
    return math.log(norm.pdf(0.5, 0, 1) + norm.pdf(0.5, 1, 2))

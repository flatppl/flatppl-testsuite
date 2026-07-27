"""Independent oracle for frag_trunc_in.

`truncate` is an UNNORMALIZED gate: inside the interval the density is just
the parent density (no renormalization by the interval mass).
"""
from scipy.stats import norm


def oracle() -> float:
    return norm.logpdf(0.5, 0, 1)

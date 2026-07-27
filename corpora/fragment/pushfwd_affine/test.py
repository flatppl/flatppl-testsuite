"""Independent oracle for frag_pushfwd_affine.

Y = 2X for X ~ Normal(0,1): the change-of-variables log-density is
`log f_X(x) - log|dy/dx|` at `x = y/2`; equals `Normal(0,2).logpdf(1.0)`
directly since scaling a Gaussian's scale by 2 is exact.
"""
import math

from scipy.stats import norm


def oracle() -> float:
    direct = norm.logpdf(0.5, 0, 1) - math.log(2)
    cross_check = norm.logpdf(1.0, 0, 2)
    assert math.isclose(direct, cross_check, rel_tol=0, abs_tol=1e-12)
    return direct

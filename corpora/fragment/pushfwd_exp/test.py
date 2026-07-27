"""Independent oracle for frag_pushfwd_exp.

Y = exp(X) for X ~ Normal(0,1) is LogNormal(s=1); the pushforward log-density
at y=1.5 is `Normal(0,1).logpdf(log y) - log y`, which is exactly
`scipy.stats.lognorm(s=1).logpdf(1.5)`.
"""
import math

from scipy.stats import lognorm, norm


def oracle() -> float:
    direct = norm.logpdf(math.log(1.5), 0, 1) - math.log(1.5)
    cross_check = lognorm.logpdf(1.5, s=1)
    assert math.isclose(direct, cross_check, rel_tol=0, abs_tol=1e-12)
    return direct

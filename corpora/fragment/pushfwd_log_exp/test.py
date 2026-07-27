"""Independent oracle for frag_pushfwd_log_exp.

Y = log(X) for X ~ Exponential(rate=1): the pushforward log-density at
y=0.5 is `logpdf_Exp(exp(y)) + y` (change of variables, Jacobian
`|dx/dy| = exp(y)`); `scipy.stats.expon.logpdf(x, scale=1) = -x`, so this is
exactly `-exp(0.5) + 0.5`.
"""
import math

from scipy.stats import expon


def oracle() -> float:
    y = 0.5
    return expon.logpdf(math.exp(y), scale=1.0) + y

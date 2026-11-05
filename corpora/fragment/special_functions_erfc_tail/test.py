"""Independent oracle for frag_special_functions_erfc_tail.

`a ~ Normal(mu = log(erfc(10)), 1)`, `b ~ Normal(mu = log(erfc(26)), 1)` (spec
§09 special-functions), scored at a = -100, b = -680. erfc(10) = 2.09e-45 and
erfc(26) = 5.66e-296, so an engine that computes 1 - erf(x) scores -inf.
f64 only: both values underflow f32.
"""
import math

from scipy.special import erfc
from scipy.stats import norm


def oracle() -> float:
    return float(norm.logpdf(-100.0, loc=math.log(erfc(10.0)), scale=1.0)
                 + norm.logpdf(-680.0, loc=math.log(erfc(26.0)), scale=1.0))

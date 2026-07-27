"""Independent oracle for ex_bayesian_inference_2.

Same model as ``ex_bayesian_inference_1`` (``bayesian_inference_2.flatppl``
is a second lowering of the identical distribution, hence identical frozen
values): ``theta1 ~ Normal(0,1)``, ``theta2 ~ Exponential(rate=1)``, ``a =
5*theta2``, ``b = abs(theta1)*theta2``, ``obs ~ iid(Normal(mu=a, sigma=b),
10)`` against the fixed 10-point ``observed_data``.
"""
from scipy.stats import expon, norm

_DATA = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]


def oracle(point: dict) -> float:
    t1, t2 = point["theta1"], point["theta2"]
    a = 5 * t2
    b = abs(t1) * t2
    lp = norm.logpdf(t1, 0, 1) + expon.logpdf(t2, scale=1.0)
    lp += sum(norm.logpdf(x, a, b) for x in _DATA)
    return float(lp)

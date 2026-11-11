"""Independent closed-form reference: a draw whose scale reads a binding derived
from two earlier draws.

v = [a, b] is a deterministic binding of the draws a ~ Exponential(rate) and
b ~ Gamma(3, 1), and y ~ Normal(0, sqrt(v[1])). With rate = 1, y is a normal whose
variance is Exponential(1), which is Laplace(0, 1/sqrt(2)). That law needs y to read
a's own draw through v, sampled once and before y; y scaled by b (variance mean 3), or
by a second draw of a, has a different law.
"""
import math

PARAMS = {"rate": 1.0}


def stat() -> dict:
    return {"distribution": {"family": "laplace", "kwargs": {"loc": 0.0, "scale": 1.0 / math.sqrt(2.0)}}, "discrete": False}

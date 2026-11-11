"""Independent closed-form reference: a draw whose scale reads a scalar binding
derived from an earlier draw.

v = 2a for a ~ Exponential(rate), and y ~ Normal(0, sqrt(v / 2)) = Normal(0, sqrt(a)).
With rate = 1, y is Laplace(0, 1/sqrt(2)), which needs y to read the same draw of a,
through v, sampled before y.
"""
import math

PARAMS = {"rate": 1.0}


def stat() -> dict:
    return {"distribution": {"family": "laplace", "kwargs": {"loc": 0.0, "scale": 1.0 / math.sqrt(2.0)}}, "discrete": False}

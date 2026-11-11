"""Independent closed-form reference: a draw conditioned on a positional `joint`.

v ~ joint(Gamma(a, 1), Gamma(3, 1)) and y ~ Normal(0, sqrt(v[1])). With a = 1,
v[1] ~ Exponential(1), and a normal whose variance is Exponential with mean 1 is
Laplace(0, 1/sqrt(2)) (variance 1). That law needs y's noise to be independent of
v[1]: noise drawn from the bits that produced v[1], or y reading the second
component (Gamma(3, 1), variance mean 3), changes the shape and the variance.
"""
import math

PARAMS = {"a": 1.0}


def stat() -> dict:
    return {"distribution": {"family": "laplace", "kwargs": {"loc": 0.0, "scale": 1.0 / math.sqrt(2.0)}}, "discrete": False}

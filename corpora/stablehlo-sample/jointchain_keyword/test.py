"""The keyword form: jointchain(a = M, b = K) calls K(record(a = a)) (spec §06).

a ~ Exponential(1) and b | a ~ Normal(0, sqrt(a)), so b is Laplace(0, 1/sqrt(2)).
That law needs b's noise to be independent of a.
"""
import math

PARAMS = {'rate': 1.0}


def stat() -> dict:
    return {"distribution": {"family": "laplace", "kwargs": {"loc": 0.0, "scale": 1.0 / math.sqrt(2.0)}}, "discrete": False}

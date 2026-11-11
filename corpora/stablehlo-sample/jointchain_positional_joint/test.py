"""The issue model: a positional joint of two Gammas as the base, then a Normal whose
variance is the first component.

z = y / sqrt(v[1]) is Normal(0, 1); y scaled by v[2] would not be. The moment
u = z^2 v[1] has mean E[v[1]] = 2 only when z is independent of v.
"""
import math

PARAMS = {'a': 2.0}


def stat() -> dict:
    return {"distribution": {"family": "norm", "kwargs": {"loc": 0.0, "scale": 1.0}}, "discrete": False, "moment": {"mean": 2.0}}

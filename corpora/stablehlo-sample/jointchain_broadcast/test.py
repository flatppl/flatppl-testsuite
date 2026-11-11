"""The issue model: an iid Gamma(2, 1) pair, then a broadcast Normal whose k-th
variance is v[k].

z = y[2] / sqrt(v[2]) is Normal(0, 1) only if y[2] reads its own v[2]; y[2]
scaled by v[1] instead is not. The moment u = z^2 v[2] has mean E[v[2]] = 2 only when
z is independent of v.
"""
import math

PARAMS = {'a': 2.0}


def stat() -> dict:
    return {"distribution": {"family": "norm", "kwargs": {"loc": 0.0, "scale": 1.0}}, "discrete": False, "moment": {"mean": 2.0}}

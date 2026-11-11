"""The issue model: a 2x3 iid Gamma(2, 1) base, then an unconditioned Normal.

The last cell v[2, 3] is Gamma(2, 1), which pins the matrix shape and its order. The
moment u = y^2 v[2, 3] has mean E[v[2, 3]] = 2 only when y is independent of v.
"""
import math

PARAMS = {'a': 2.0}


def stat() -> dict:
    return {"distribution": {"family": "gamma", "kwargs": {"a": 2.0, "scale": 1.0}}, "discrete": False, "moment": {"mean": 2.0}}

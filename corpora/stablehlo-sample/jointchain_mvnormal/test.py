"""The issue model: a Beta base, then an MvNormal whose covariance is diagmat([1, s]).

y[2] | s ~ Normal(0, sqrt(s)), so z = y[2] / sqrt(s) is Normal(0, 1) for every s.
The moment u = z^2 s has mean E[s] = 2/5 = 0.4 only when z is independent of s.
"""
import math

PARAMS = {'a': 2.0}


def stat() -> dict:
    return {"distribution": {"family": "norm", "kwargs": {"loc": 0.0, "scale": 1.0}}, "discrete": False, "moment": {"mean": 0.4}}

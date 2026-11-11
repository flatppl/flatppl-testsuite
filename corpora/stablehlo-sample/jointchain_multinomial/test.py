"""The issue model: a Multinomial base, then a Normal centred on the first count.

z = y - n[1] is Normal(0, 1) for every n. The moment u = z^2 n[1] has mean
E[n[1]] = 10 * 0.2 = 2 only when z is independent of n.
"""
import math

PARAMS = {'p': [0.2, 0.3, 0.5]}


def stat() -> dict:
    return {"distribution": {"family": "norm", "kwargs": {"loc": 0.0, "scale": 1.0}}, "discrete": False, "moment": {"mean": 2.0}}

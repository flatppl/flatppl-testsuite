"""The base marginal of the MvNormal chain: s ~ Beta(2, 3).
"""
import math

PARAMS = {'a': 2.0}


def stat() -> dict:
    return {"distribution": {"family": "beta", "kwargs": {"a": 2.0, "b": 3.0}}, "discrete": False}

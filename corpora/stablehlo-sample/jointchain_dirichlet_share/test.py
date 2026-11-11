"""The base marginal of the Dirichlet chain: v[1] of Dirichlet([2, 3, 5]) is Beta(2, 3 + 5).
"""
import math

PARAMS = {'alpha': [2.0, 3.0, 5.0]}


def stat() -> dict:
    return {"distribution": {"family": "beta", "kwargs": {"a": 2.0, "b": 8.0}}, "discrete": False}

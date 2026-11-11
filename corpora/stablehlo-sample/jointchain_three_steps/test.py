"""A chain of three: the third kernel reads the record of both earlier fields.

c is Laplace(0, 1/sqrt(2)) when c reads the base a. b and c are independent given a,
so u = b c has mean 0; c drawn from b's key would give E[u] = E[a] = 1.
"""
import math

PARAMS = {'rate': 1.0}


def stat() -> dict:
    return {"distribution": {"family": "laplace", "kwargs": {"loc": 0.0, "scale": 1.0 / math.sqrt(2.0)}}, "discrete": False, "moment": {"mean": 0.0}}

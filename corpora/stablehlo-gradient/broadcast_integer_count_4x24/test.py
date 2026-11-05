"""Numpy closed form for an integer running count mapped over rows.

H[r, t] is the count of k[r, 0..t]. The output is log(a) * sum(H) - a * sum(H^2),
so d/da is sum(H) / a - sum(H^2).
"""

import numpy as np


def _h(point):
    return np.cumsum(np.asarray(point["k"], dtype=np.int64), axis=1)


def oracle(point):
    a, h = float(point["a"]), _h(point)
    return float(np.log(a) * np.sum(h) - a * np.sum(h * h))


def grad_oracle(point):
    a, h = float(point["a"]), _h(point)
    return {"a": float(np.sum(h) / a - np.sum(h * h))}

"""Independent inverse-metric quadratic, trace and raising identity."""

import numpy as np


def oracle(point):
    g = np.asarray(point["metric_entries"], dtype=float).reshape(2, 2)
    v = np.asarray(point["v"], dtype=float)
    a = np.array([[1., 2.], [3., 4.]])
    inverse = np.linalg.inv(g)
    return float(v @ inverse @ v + np.trace(a @ inverse) + 2*v.sum())

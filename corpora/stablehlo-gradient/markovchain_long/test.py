"""Closed-form derivatives of a Gaussian random walk with drift.

With increments d_k = x_k - x_{k-1} (x_0 = 0) and r_k = d_k - drift,
d/d drift = sum r_k / s^2 and d/d scale = sum (r_k^2 / s^3 - 1 / s).
"""

import numpy as np

T = np.linspace(0.0, 20.0, 1000)
X = np.sin(T) + 0.05 * T
D = np.diff(X, prepend=0.0)


def grad_oracle(point):
    drift, scale = point["drift"], point["scale"]
    r = D - drift
    return {
        "drift": float(np.sum(r) / scale**2),
        "scale": float(np.sum(r**2 / scale**3 - 1.0 / scale)),
    }

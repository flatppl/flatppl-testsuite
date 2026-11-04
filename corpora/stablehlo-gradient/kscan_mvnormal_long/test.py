"""Closed-form derivatives of a Gaussian vector autoregression.

With prev_k = x_{k-1} (x_0 = 0.5 * 1), residual r_k = x_k - phi prev_k - a u_k 1
and P = Sigma^-1, d/d phi = sum prev_k' P r_k and d/d a = sum u_k 1' P r_k.
"""

import numpy as np

T = np.linspace(0.0, 20.0, 1000)
U = np.linspace(-1.0, 1.0, 1000)
X = np.stack([np.sin(T), np.cos(T), np.sin(2.0 * T)], axis=1)
PREV = np.vstack([np.full(3, 0.5), X[:-1]])
P = np.linalg.inv(np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.3], [0.0, 0.3, 1.0]]))


def grad_oracle(point):
    phi, a = point["phi"], point["a"]
    r = X - phi * PREV - a * U[:, None]
    w = r @ P
    return {
        "phi": float(np.sum(PREV * w)),
        "a": float(np.sum(U * w.sum(axis=1))),
    }

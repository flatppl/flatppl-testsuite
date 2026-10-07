"""Differentiate log determinants and trace(V^-1 X) directly."""

import numpy as np


def grad_oracle(point):
    inverse_scale = np.linalg.inv(point["scale"])
    x = np.asarray(point["x"])
    return {
        "scale": ((inverse_scale @ x @ inverse_scale - 5 * inverse_scale) / 2).tolist(),
        "x": (np.linalg.inv(x) - inverse_scale / 2).tolist(),
    }

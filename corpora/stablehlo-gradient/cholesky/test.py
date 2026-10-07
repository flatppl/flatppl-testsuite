"""Symmetric Cholesky differential with an asymmetric output cotangent."""

import numpy as np


def grad_oracle(point):
    lower = np.linalg.cholesky(point["A"])
    weights = np.array([[1.0, 0.0], [2.0, 3.0]])
    phi = np.tril(lower.T @ weights)
    phi[np.diag_indices(2)] *= 0.5
    inverse = np.linalg.inv(lower)
    gradient = inverse.T @ phi @ inverse
    return {"A": ((gradient + gradient.T) / 2).tolist()}

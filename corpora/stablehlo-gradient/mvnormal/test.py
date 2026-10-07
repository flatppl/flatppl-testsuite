"""Closed-form mean, covariance and point derivatives of a Gaussian density."""

import numpy as np


def grad_oracle(point):
    covariance = np.asarray(point["cov"])
    residual = np.asarray(point["x"]) - point["mu"]
    alpha = np.linalg.solve(covariance, residual)
    gradient = (np.outer(alpha, alpha) - np.linalg.inv(covariance)) / 2
    return {"mu": alpha.tolist(), "cov": gradient.tolist(), "x": (-alpha).tolist()}

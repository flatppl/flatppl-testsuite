"""Independent oracle for frag_pushfwd_affine_vector_mvnormal.

Y = a .* X .+ b for X ~ MvNormal(mu, cov): the base density stays coupled and
the Jacobian is diag(a), so log p(y) = log N((y - b) ./ a; mu, cov) - Σᵢ log|aᵢ|.
Y is also MvNormal(a .* mu .+ b, D cov D) with D = diag(a), a second derivation.
"""
import math

import numpy as np
from scipy.stats import multivariate_normal


def oracle() -> float:
    a = np.array([2.0, -0.4, 1.5])
    b = np.array([0.3, -1.0, 2.0])
    mu = np.array([0.2, -0.1, 0.4])
    cov = np.array([[1.0, 0.3, 0.1], [0.3, 2.0, -0.4], [0.1, -0.4, 1.5]])
    y = np.array([0.7, -1.2, 2.5])
    direct = float(multivariate_normal.logpdf((y - b) / a, mu, cov) - np.sum(np.log(np.abs(a))))
    d = np.diag(a)
    cross_check = float(multivariate_normal.logpdf(y, d @ mu + b, d @ cov @ d))
    assert math.isclose(direct, cross_check, rel_tol=0, abs_tol=1e-12)
    return direct

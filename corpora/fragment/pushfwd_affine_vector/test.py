"""Independent oracle for frag_pushfwd_affine_vector.

Y = a .* X .+ b for X ~ iid Normal(0, 1) is the elementwise affine map with
the diagonal Jacobian diag(a), so log p(y) = Σᵢ [log φ((yᵢ - bᵢ)/aᵢ) - log|aᵢ|].
Each Yᵢ is also Normal(bᵢ, |aᵢ|) directly, a second derivation.
"""
import math

import numpy as np
from scipy.stats import norm


def oracle() -> float:
    a = np.array([2.0, -0.4, 1.5])
    b = np.array([0.3, -1.0, 2.0])
    y = np.array([0.7, -1.2, 2.5])
    direct = float(np.sum(norm.logpdf((y - b) / a) - np.log(np.abs(a))))
    cross_check = float(np.sum(norm.logpdf(y, b, np.abs(a))))
    assert math.isclose(direct, cross_check, rel_tol=0, abs_tol=1e-12)
    return direct

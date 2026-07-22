"""Independent scipy oracle for the linear_regression posterior logdensity.

Posterior is UNNORMALIZED (bayesupdate = prior x likelihood):
  log post(alpha,beta,sigma)
    = log p(sigma)                       # sqrt-pushforward of InverseGamma(5,5)
    + log N(alpha; 0, sigma*3)
    + log N(beta;  0, sigma*3)
    + sum_i log N(y_i; alpha + beta*x_i, sigma)

p(sigma): X ~ InverseGamma(a=5, scale=5), S = sqrt(X) =>
  log p_S(s) = InvGamma.logpdf(s**2; a=5, scale=5) + log(2*s).
scipy.stats.invgamma(a, scale=scale) has pdf x^{-a-1} exp(-scale/x) scale^a / Gamma(a),
matching FlatPPL InverseGamma(alpha=5, beta=5) with shape=5, scale(beta)=5.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

X_DATA = np.array([1.1, 1.5, 1.3, 1.4])
Y_DATA = np.array([3.2, 4.1, 3.4, 3.9])
IG_A = 5.0
IG_SCALE = 5.0


def oracle(point: dict) -> float:
    a = point["alpha_v"]
    b = point["beta_v"]
    s = point["sigma_v"]
    # prior over sigma: sqrt-pushforward of InverseGamma(5,5)
    lp_sigma = stats.invgamma.logpdf(s * s, IG_A, scale=IG_SCALE) + math.log(2.0 * s)
    lp_alpha = stats.norm.logpdf(a, loc=0.0, scale=s * 3.0)
    lp_beta = stats.norm.logpdf(b, loc=0.0, scale=s * 3.0)
    means = a + b * X_DATA
    lp_like = float(np.sum(stats.norm.logpdf(Y_DATA, loc=means, scale=s)))
    return float(lp_sigma + lp_alpha + lp_beta + lp_like)

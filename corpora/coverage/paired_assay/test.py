"""Independent oracle for cov_paired_assay (scipy).

The table variate factorizes row-wise and field-wise (§06 `iid` over a
record law; the components share no stochastic node given mu_a, mu_b):

    lp = sum_i Normal(a_i | mu_a, 1).logpdf
       + sum_i Normal(b_i | mu_b, 2).logpdf
       + Normal(mu_a | 0, 5).logpdf + Normal(mu_b | 0, 5).logpdf

STATUS: scored live since 2026-09-01. The determiniser gained the
record-measure unroll (the table row count comes from the iid node's
own `%table` domain, and each row is read out of its columns), and the
same change stopped the keyword-`joint` record-law rewrite from
marginalizing a latent that the enclosing `kernelof` declares as an
INPUT — §06 `likelihoodof` makes `densityof(likelihoodof(K, obs),
theta)` the CONDITIONAL `pdf(K(theta), obs)`. Before that the obs were
scored against the prior predictive, Normal(0, sqrt(26)) /
Normal(0, sqrt(29)), which is -36.27 at the first point rather than
-22.89.
"""
import numpy as np
from scipy import stats

A = np.array([0.61, -0.32, 1.4, 0.05, -0.88, 0.97])
B = np.array([1.1, 2.53, -0.4, 0.75, 1.9, -0.21])


def oracle(point: dict) -> float:
    mu_a, mu_b = float(point["mu_a"]), float(point["mu_b"])
    lik = stats.norm.logpdf(A, mu_a, 1.0).sum() + stats.norm.logpdf(B, mu_b, 2.0).sum()
    prior = stats.norm.logpdf(mu_a, 0.0, 5.0) + stats.norm.logpdf(mu_b, 0.0, 5.0)
    return float(lik + prior)

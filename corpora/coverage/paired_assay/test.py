"""Independent oracle for cov_paired_assay (scipy).

The table variate factorizes row-wise and field-wise (§06 `iid` over a
record law; the components share no stochastic node given mu_a, mu_b):

    lp = sum_i Normal(a_i | mu_a, 1).logpdf
       + sum_i Normal(b_i | mu_b, 2).logpdf
       + Normal(mu_a | 0, 5).logpdf + Normal(mu_b | 0, 5).logpdf

STATUS: the rust determiniser REFUSES the model at the `iid` node even
with a LITERAL size 6 — "iid size is not a statically-resolved 1-D
count (dynamic, multi-axis, or unresolved domain)" — i.e. the
record-valued inner measure (table variate) has no unroll arm
(verified live 2026-09-01; a scalar `iid(Normal, 3)` lowers in the
same setup, so the size is not the operative part of the message).
`allow_skip: true`; the frozen values take over when tables lower.
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

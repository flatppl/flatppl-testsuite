"""Independent scipy oracle for cov_mv_mixture: a two-component mixture of
BIVARIATE normals, scored at five observation points.

    lp(y) = logsumexp_i(log w_i + mvn(mu_i, cov_i).logpdf(y)) - log(sum w)

which is §06 `ksuperpose`'s own density rule. Sum w = 1 here, so the
`normalize` divisor is 1 and the last term vanishes.

STATUS: the rust determiniser REFUSES the model — "ksuperpose over a
MULTIVARIATE parameter family is not lowered (the per-component slice
extraction is not built): a family argument with an axis beyond the family
axis has no `broadcast(record, ...)` form" (observed live 2026-09-01). §04
*Collection arguments* makes a plain broadcast require the same number of
axes from every collection argument and strips every axis to reach its
cell, so an N x d `mu` beside an N x d x d `cov` has no form there.
`allow_skip: true`; the frozen values take over when the lowering lands.

The flatppl-js engine scores this same mixture correctly today (its
`ksuperpose` expansion indexes only the family axis), verified against
these values in
`flatppl-js/packages/engine/test/ksuperpose-multivariate.test.ts`.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

W = np.array([0.3, 0.7])
MUS = np.array([[-2.0, 1.0], [3.0, -1.0]])
COVS = [
    np.array([[1.0, 0.3], [0.3, 1.0]]),
    np.array([[0.5, 0.0], [0.0, 2.0]]),
]


def oracle(point: dict) -> float:
    y = np.asarray(point["y"], dtype=float)
    terms = [
        np.log(w) + multivariate_normal(mean=m, cov=c).logpdf(y)
        for w, m, c in zip(W, MUS, COVS)
    ]
    return float(logsumexp(terms) - np.log(W.sum()))

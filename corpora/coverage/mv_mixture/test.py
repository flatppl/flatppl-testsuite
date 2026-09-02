"""Independent scipy oracle for cov_mv_mixture: a two-component mixture of
BIVARIATE normals, scored at five observation points.

    lp(y) = logsumexp_i(log w_i + mvn(mu_i, cov_i).logpdf(y)) - log(sum w)

which is §06 `ksuperpose`'s own density rule. Sum w = 1 here, so the
`normalize` divisor is 1 and the last term vanishes.

STATUS: scored. The rust determiniser lowers the multivariate family to a
per-component slice form — get(mus, i, all) for the flat N x d `mu`, and
get(covs, i) for the nested N-vector of matrices — then the §06 mixture
assembly, one `builtin_logdensityof` per component under one `logsumexp`
(2026-09-02). It refused before that: a plain `broadcast(record, ...)`
requires the same number of axes from every collection argument (§04
*Collection arguments*) and strips every axis to reach its cell, so an
N x d `mu` beside an N x d x d `cov` has no form there.

The flatppl-js engine reaches the same slice rule independently, in
`flatppl-js/packages/engine/ksuperpose-expand.ts`. That agreement is not
evidence: the values below are scipy's.

NO StableHLO row: `crates/stablehlo` still refuses the slice form the
determiniser now emits. Re-probed live 2026-09-02 at rust `b08eb79` with
this dir's own `query.flatppl` — `flatppl stablehlo` exits 3 with
"stablehlo: expected 2 argument(s), got 3", from `ops.rs`'s `lower_get`,
which has no multi-selector `get` arm for `get(mus, i, all)`. The
determiniser refusal is gone; the emitter gap is not.
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

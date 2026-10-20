"""Independent oracle for frag_shared_latent_joint.

The shared ancestor in `z ~ Normal(mu=0.0, sigma=1.0)` and
`y1, y2 | z ~ Normal(mu=z, sigma=1.0)` is retained inside one
`lawof(record(y1 = y1, y2 = y2))`. Separate law components in `joint` would
instead be independent. The law of `record(y1 = y1, y2 = y2)` is
`MvNormal(mu0 * 1, s0**2 * J + diag(sigma**2))` with `J` all-ones, scored at
`(y1, y2) = (0.5, 0.7)` -- the identical model and point as
`shared_latent_record`, so it freezes the SAME expected value.

scipy's `multivariate_normal` shares no algebra with the emitted expression:
the determiniser lowers the record law through Sherman-Morrison and the
matrix determinant lemma.
"""
from scipy.stats import multivariate_normal

MU0 = 0.0
S0 = 1.0
SIGMA = (1.0, 1.0)
POINT = (0.5, 0.7)


def oracle() -> float:
    n = len(SIGMA)
    cov = [[S0**2 + (SIGMA[i] ** 2 if i == j else 0.0) for j in range(n)]
           for i in range(n)]
    return float(multivariate_normal(mean=[MU0] * n, cov=cov).logpdf(POINT))

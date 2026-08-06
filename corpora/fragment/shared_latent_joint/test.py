"""Independent oracle for frag_shared_latent_joint.

The `joint` spelling of `shared_latent_record`: `joint(y1 = lawof(y1), y2 =
lawof(y2))` over `z ~ Normal(mu=0.0, sigma=1.0)`; `y1, y2 | z ~ Normal(mu=z,
sigma=1.0)`. §06 "Equivalent record law" makes this joint
equivalent to `lawof(record(y1 = y1, y2 = y2))` -- the shared ancestor `z` is
traced once, so the two named components stay correlated rather than becoming
independent copies. The joint law of `record(y1 = y1, y2 = y2)` is
`MvNormal(mu0 * 1, s0**2 * J + diag(sigma**2))` with `J` all-ones, scored at
`(y1, y2) = (0.5, 0.7)` -- the identical model and point as
`shared_latent_record`, so it freezes the SAME expected value.

scipy's `multivariate_normal` shares no algebra with the emitted expression:
the determiniser lowers the `joint` rewrite through the same Sherman-Morrison
and matrix-determinant-lemma path as the plain record spelling.
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

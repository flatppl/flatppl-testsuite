"""Independent oracle for frag_shared_latent_record.

`z ~ Normal(mu=0.0, sigma=1.0)`; `y1, y2 | z ~ Normal(mu=z, sigma=1.0)`. The two
fields are correlated through `z`, so the joint law of `record(y1 = y1, y2 = y2)`
is `MvNormal(mu0 * 1, s0**2 * J + diag(sigma**2))` with `J` all-ones, scored at
`(y1, y2) = (0.5, 0.7)`.

The product of the two correct `Normal(0, sqrt(2))` marginals is
-2.716024246969291, which is what the determiniser emitted for this shape before
flatppl-rust #131 — 0.199 nats away, so this point discriminates.

scipy's `multivariate_normal` shares no algebra with the emitted expression: the
determiniser lowers the same density through Sherman-Morrison and the matrix
determinant lemma as scalar §07 ops. Cross-checked against Gauss-Kronrod
quadrature of `int phi(z) N(0.5; z, 1) N(0.7; z, 1) dz`, which agrees to 4.4e-16.
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

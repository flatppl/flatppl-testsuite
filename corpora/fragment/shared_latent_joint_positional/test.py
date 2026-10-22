"""Independent oracle for frag_shared_latent_joint_positional.

The vector-input spelling of `shared_latent_joint`: the query maps the
coordinates of `[0.5, 0.7]` into the named fields of one record law.
Both outputs retain their shared latent inside that law. Separate law
components in `joint` would instead be independent. The model and point
retain the frozen value from `shared_latent_record`:
`MvNormal(mu0 * 1, s0**2 * J + diag(sigma**2))` with `J` all-ones,
at `(y1, y2) = (0.5, 0.7)`.
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

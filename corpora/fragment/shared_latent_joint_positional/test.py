"""Independent oracle for frag_shared_latent_joint_positional.

The POSITIONAL spelling of `shared_latent_joint`: §06 "Joint composition"
makes the positional form of `joint` "the corresponding `cat` law", so
`joint(lawof(y1), lawof(y2))` reaches the same shared-latent record-law
machinery as the keyword spelling, via synthetic field names and a value
record sliced from the flat `cat` variate `[0.5, 0.7]`. Same model, same
point, so the same frozen value as `shared_latent_joint` and
`shared_latent_record`: `MvNormal(mu0 * 1, s0**2 * J + diag(sigma**2))` with
`J` all-ones, at `(y1, y2) = (0.5, 0.7)`.
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

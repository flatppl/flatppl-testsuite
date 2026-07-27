"""Independent oracle for frag_gamma_poisson_marginal.

`z ~ Gamma(shape=2.0, rate=3.0)`, `y | z ~ Poisson(rate=z)`; the Gamma-Poisson
conjugate marginal of y is `NegativeBinomial(alpha=2.0, beta=3.0)` (§08),
scored at y=4. scipy's `nbinom(n, p)` pmf is `C(k+n-1, k) p^n (1-p)^k`;
matching against the §08 pmf `C(k+alpha-1, alpha-1) (beta/(beta+1))^alpha
(1/(beta+1))^k` gives n=alpha, p=beta/(beta+1) (checked by hand: both give
-4.511103676949024 for alpha=2, beta=3, k=4).
"""
from scipy.stats import nbinom


def oracle() -> float:
    alpha, beta = 2.0, 3.0
    return nbinom.logpmf(4, n=alpha, p=beta / (beta + 1.0))

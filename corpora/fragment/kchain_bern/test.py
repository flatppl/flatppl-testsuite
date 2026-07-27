"""Independent oracle for frag_kchain_bern.

`z ~ Bernoulli(p=0.3)`, `y | z ~ Normal(mu=z, sigma=1)`; the marginal of y at
1.5 is the Bernoulli-weighted 2-component mixture
`(1-p) N(1.5; 0, 1) + p N(1.5; 1, 1)`.
"""
import math

from scipy.stats import norm


def oracle() -> float:
    p = 0.3
    return math.log((1 - p) * norm.pdf(1.5, 0, 1) + p * norm.pdf(1.5, 1, 1))

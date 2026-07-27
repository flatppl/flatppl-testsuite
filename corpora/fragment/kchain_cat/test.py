"""Independent oracle for frag_kchain_cat.

`z ~ Categorical(p=[0.2,0.3,0.5])` over 1-based atoms {1,2,3}, `y | z ~
Normal(mu=z, sigma=1)`; the marginal of y at 0.5 is the Categorical-weighted
3-component mixture.
"""
import math

from scipy.stats import norm


def oracle() -> float:
    weights = [0.2, 0.3, 0.5]
    mus = [1, 2, 3]
    return math.log(sum(w * norm.pdf(0.5, mu, 1) for w, mu in zip(weights, mus)))

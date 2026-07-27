"""Independent oracle for frag_broadcast_poisson.

`broadcast(Poisson, [2.0, 3.5, 1.0])` is an array-of-kernels measure over the
length-3 observation array (spec §04 broadcasting); its log-density at
`[1, 4, 2]` is the SUM of independent per-cell Poisson log-pmfs,
`Σᵢ Poisson.logpmf(kᵢ; λᵢ)` for λ=[2.0, 3.5, 1.0], k=[1, 4, 2].
"""
from scipy.stats import poisson


def oracle() -> float:
    lambdas = [2.0, 3.5, 1.0]
    ks = [1, 4, 2]
    return sum(poisson.logpmf(k, lam) for k, lam in zip(ks, lambdas))

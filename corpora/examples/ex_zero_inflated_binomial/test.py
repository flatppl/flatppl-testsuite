"""Independent oracle for ex_zero_inflated_binomial.

``p ~ Beta(1.5,1.5)`` (success prob given not zero-inflated), ``psi ~
Beta(1.5,1.5)`` (mixing weight); ``ZeroInflatedBinomial = superpose(
weighted(psi, Binomial(K,p)), weighted(1-psi, Dirac(0)))`` -- not wrapped in
``normalize`` here, but it's already a proper measure (weights sum to 1, both
mixands have total mass 1); ``y ~ iid(ZeroInflatedBinomial, 10)`` against the
fixed ``y_obs`` (``K=20`` trials each). At ``y == 0`` both mixands
contribute (``Dirac(0)`` has log-density 0 there); at ``y != 0`` only the
Binomial branch does (``Dirac(0)``'s log-density is ``-inf``).
"""
import math

from scipy.special import logsumexp
from scipy.stats import beta as beta_dist, binom

_K = 20
_Y_OBS = [7, 0, 5, 8, 0, 6, 4, 0, 9, 3]


def oracle(point: dict) -> float:
    p, psi = point["p"], point["psi"]
    lp = beta_dist.logpdf(p, 1.5, 1.5) + beta_dist.logpdf(psi, 1.5, 1.5)
    for y in _Y_OBS:
        log_binom_branch = math.log(psi) + binom.logpmf(y, _K, p)
        log_dirac_branch = math.log(1.0 - psi) + (0.0 if y == 0 else -math.inf)
        lp += logsumexp([log_binom_branch, log_dirac_branch])
    return float(lp)

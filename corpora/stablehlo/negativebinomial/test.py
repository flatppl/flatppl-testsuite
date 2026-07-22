"""Independent scipy oracle: NegativeBinomial(alpha, beta) logdensity at the
fixed observed a=4.

FlatPPL NegativeBinomial(alpha, beta): pmf = C(k+alpha-1,alpha-1) *
(beta/(beta+1))^alpha * (1/(beta+1))^k, matched to scipy's
nbinom.pmf(k,n,p) = C(k+n-1,k) p^n (1-p)^k via n=alpha, p=beta/(beta+1).
"""
from scipy.stats import nbinom


def oracle(point: dict) -> float:
    alpha = point["alpha"]
    beta = point["beta"]
    return float(nbinom.logpmf(4, alpha, beta / (beta + 1.0)))

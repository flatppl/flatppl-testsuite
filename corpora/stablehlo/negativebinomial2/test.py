"""Independent scipy oracle: NegativeBinomial2(mu, psi) logdensity at the
fixed observed a=4.

FlatPPL NegativeBinomial2(mu, psi): pmf = C(k+psi-1,k) * (mu/(mu+psi))^k *
(psi/(mu+psi))^psi, matched to scipy's nbinom.pmf(k,n,p) via n=psi,
p=psi/(mu+psi).
"""
from scipy.stats import nbinom


def oracle(point: dict) -> float:
    mu = point["mu"]
    psi = point["psi"]
    return float(nbinom.logpmf(4, psi, psi / (mu + psi)))

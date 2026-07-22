"""Independent analytic oracle: gradient of NegativeBinomial2(mu, psi)
logdensity (mean/dispersion parametrisation) at the fixed variate k=4 (see
query.flatppl) w.r.t. mu and psi.
d/dmu = k/mu - (psi+k)/(mu+psi)
d/dpsi = digamma(k+psi) - digamma(psi) + log(psi/(mu+psi)) + 1
         - psi/(mu+psi) - k/(mu+psi)."""
import math

from scipy.special import digamma

_K = 4


def grad_oracle(point: dict) -> dict:
    mu = point["mu"]
    psi = point["psi"]
    dmu = _K / mu - (psi + _K) / (mu + psi)
    dpsi = (
        digamma(_K + psi) - digamma(psi) + math.log(psi / (mu + psi)) + 1.0
        - psi / (mu + psi) - _K / (mu + psi)
    )
    return {"mu": dmu, "psi": dpsi}

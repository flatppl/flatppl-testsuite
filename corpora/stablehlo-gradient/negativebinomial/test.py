"""Independent analytic oracle: gradient of NegativeBinomial(alpha, beta)
logdensity at the fixed variate k=4 (see query.flatppl) w.r.t. alpha and beta.
logpdf = lgamma(k+alpha) - lgamma(alpha) - lgamma(k+1) + alpha*log(beta/(beta+1))
- k*log(beta+1), so
d/dalpha = digamma(k+alpha) - digamma(alpha) + log(beta/(beta+1))
d/dbeta = alpha*(1/beta - 1/(beta+1)) - k/(beta+1)."""
import math

from scipy.special import digamma

_K = 4


def grad_oracle(point: dict) -> dict:
    alpha = point["alpha"]
    beta = point["beta"]
    dalpha = digamma(_K + alpha) - digamma(alpha) + math.log(beta / (beta + 1.0))
    dbeta = alpha * (1.0 / beta - 1.0 / (beta + 1.0)) - _K / (beta + 1.0)
    return {"alpha": dalpha, "beta": dbeta}

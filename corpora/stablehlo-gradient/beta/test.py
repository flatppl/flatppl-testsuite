"""Independent analytic oracle: gradient of Beta(alpha, beta) logdensity at
xobs w.r.t. alpha and beta. logpdf = (alpha-1)*log(x) + (beta-1)*log(1-x)
- lbeta(alpha, beta), so d/dalpha = log(x) - digamma(alpha) + digamma(alpha+beta)
and d/dbeta = log(1-x) - digamma(beta) + digamma(alpha+beta)."""
import math

from scipy.special import digamma


def grad_oracle(point: dict) -> dict:
    alpha = point["alpha"]
    beta = point["beta"]
    x = point["xobs"]
    dab = digamma(alpha + beta)
    dalpha = math.log(x) - digamma(alpha) + dab
    dbeta = math.log(1.0 - x) - digamma(beta) + dab
    return {"alpha": dalpha, "beta": dbeta}

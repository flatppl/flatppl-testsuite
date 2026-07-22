"""Independent analytic oracle: gradient of Dirichlet(alpha) logdensity at the
fixed variate x=[0.2, 0.3, 0.5] (see query.flatppl) w.r.t. each component of
alpha. logpdf = lgamma(sum(alpha)) - sum(lgamma(alpha_i)) + sum((alpha_i-1)*log(x_i)),
so d/dalpha_i = digamma(sum(alpha)) - digamma(alpha_i) + log(x_i)."""
import math

from scipy.special import digamma

_X = [0.2, 0.3, 0.5]


def grad_oracle(point: dict) -> dict:
    alpha = point["alpha"]
    total = sum(alpha)
    dsum = digamma(total)
    grad = [dsum - digamma(a) + math.log(x) for a, x in zip(alpha, _X)]
    return {"alpha": grad}

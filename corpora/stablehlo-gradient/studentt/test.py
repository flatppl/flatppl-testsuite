"""Independent analytic oracle: gradient of StudentT(nu) logdensity at xobs
w.r.t. nu. logpdf = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(nu*pi)
- (nu+1)/2*log(1+x**2/nu), so
d/dnu = 0.5*(digamma((nu+1)/2) - digamma(nu/2)) - 1/(2*nu)
        - 0.5*log(1+x**2/nu) + (nu+1)/2*(x**2/nu**2)/(1+x**2/nu)."""
import math

from scipy.special import digamma


def grad_oracle(point: dict) -> dict:
    nu = point["nu"]
    x = point["xobs"]
    ratio = x**2 / nu
    dnu = (
        0.5 * (digamma((nu + 1) / 2) - digamma(nu / 2))
        - 1.0 / (2 * nu)
        - 0.5 * math.log(1.0 + ratio)
        + (nu + 1) / 2 * (x**2 / nu**2) / (1.0 + ratio)
    )
    return {"nu": dnu}

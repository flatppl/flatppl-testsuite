"""Independent analytic oracle: gradient of LogNormal(mu, sigma) logdensity at
xobs w.r.t. mu and sigma. logpdf = -log(x) - log(sigma) - 0.5*log(2*pi)
- (log(x)-mu)**2/(2*sigma**2), so d/dmu = (log(x)-mu)/sigma**2 and
d/dsigma = -1/sigma + (log(x)-mu)**2/sigma**3."""
import math


def grad_oracle(point: dict) -> dict:
    mu = point["mu"]
    sigma = point["sigma"]
    x = point["xobs"]
    dmu = (math.log(x) - mu) / sigma**2
    dsigma = -1.0 / sigma + (math.log(x) - mu) ** 2 / sigma**3
    return {"mu": dmu, "sigma": dsigma}

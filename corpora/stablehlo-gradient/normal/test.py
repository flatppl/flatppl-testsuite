"""Independent analytic oracle: gradient of Normal(mu, sigma) logdensity at
xobs w.r.t. mu and sigma. logpdf = -0.5*log(2*pi) - log(sigma) - (x-mu)**2/(2*sigma**2),
so d/dmu = (x-mu)/sigma**2 and d/dsigma = -1/sigma + (x-mu)**2/sigma**3."""


def grad_oracle(point: dict) -> dict:
    mu = point["mu"]
    sigma = point["sigma"]
    x = point["xobs"]
    dmu = (x - mu) / sigma**2
    dsigma = -1.0 / sigma + (x - mu) ** 2 / sigma**3
    return {"mu": dmu, "sigma": dsigma}

"""Independent analytic oracle: gradient of Normal(mu = digamma(m), sigma = 1)
logdensity at a = 0.5 w.r.t. m: (a - mu) * polygamma(1, m) (spec §09 special-functions)."""
from scipy import special


def grad_oracle(point: dict) -> dict:
    m = point["m"]
    return {"m": float((0.5 - special.digamma(m)) * special.polygamma(1, m))}

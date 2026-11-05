"""Independent analytic oracle: gradient of Normal(mu = erfc(m), sigma = 1)
logdensity at a = 0.5 w.r.t. m. With mu = erfc(m), d/dm = (a - mu) * dmu/dm and
dmu/dm = -2/sqrt(pi)*exp(-m**2) (spec §09 special-functions)."""
import math

from scipy import special


def grad_oracle(point: dict) -> dict:
    m = point["m"]
    mu = special.erfc(m)
    dmu = -2.0 / math.sqrt(math.pi) * math.exp(-m * m)
    return {"m": float((0.5 - mu) * dmu)}

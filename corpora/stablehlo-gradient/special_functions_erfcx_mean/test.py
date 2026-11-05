"""Independent analytic oracle: gradient of Normal(mu = erfcx(m), sigma = 1)
logdensity at a = 0.5 w.r.t. m. d/dm = (a - mu) * dmu/dm with
dmu/dm = 2 m erfcx(m) - 2/sqrt(pi) (spec §09 special-functions)."""
import math

from scipy import special


def grad_oracle(point: dict) -> dict:
    m = point["m"]
    mu = special.erfcx(m)
    dmu = 2.0 * m * mu - 2.0 / math.sqrt(math.pi)
    return {"m": float((0.5 - mu) * dmu)}

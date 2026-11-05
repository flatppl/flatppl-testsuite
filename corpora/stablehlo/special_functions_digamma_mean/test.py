"""Independent scipy oracle: Normal(mu = digamma(m), sigma = 1) logdensity at a = 0.5
(spec §09 special-functions)."""
from scipy import special, stats


def oracle(point: dict) -> float:
    m = point["m"]
    return float(stats.norm.logpdf(0.5, loc=special.digamma(m), scale=1.0))

"""Independent scipy oracle: Normal(mu = erfc(m), sigma = 1) logdensity at
a = 0.5 (spec §09 special-functions)."""
from scipy import special, stats


def oracle(point: dict) -> float:
    return float(stats.norm.logpdf(0.5, loc=special.erfc(point["m"]), scale=1.0))

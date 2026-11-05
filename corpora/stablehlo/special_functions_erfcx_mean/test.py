"""Independent scipy oracle: Normal(mu = erfcx(m), sigma = 1) logdensity at
a = 0.5 (spec §09 special-functions, erfcx(x) = exp(x^2) erfc(x))."""
from scipy import special, stats


def oracle(point: dict) -> float:
    return float(stats.norm.logpdf(0.5, loc=special.erfcx(point["m"]), scale=1.0))

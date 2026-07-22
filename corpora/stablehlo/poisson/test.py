"""Independent scipy oracle: Poisson(rate) logdensity at the fixed observed a=3."""
from scipy.stats import poisson


def oracle(point: dict) -> float:
    return float(poisson.logpmf(3, point["rate"]))

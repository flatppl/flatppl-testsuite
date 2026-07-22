"""Independent scipy oracle: Bernoulli(p) logdensity at the fixed observed a=1."""
from scipy.stats import bernoulli


def oracle(point: dict) -> float:
    return float(bernoulli.logpmf(1, point["p"]))

"""Independent scipy oracle: Binomial(n, p) logdensity at the fixed observed a=2."""
from scipy.stats import binom


def oracle(point: dict) -> float:
    return float(binom.logpmf(2, int(round(point["n"])), point["p"]))

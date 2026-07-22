"""Independent scipy oracle: Categorical0(p) [0-based] logdensity at the fixed
observed a=1."""
from scipy.stats import rv_discrete


def oracle(point: dict) -> float:
    p = point["p"]
    return float(rv_discrete(values=([0, 1, 2], p)).logpmf(1))

"""Independent scipy oracle: Categorical(p) [1-based] logdensity at the fixed
observed a=2."""
from scipy.stats import rv_discrete


def oracle(point: dict) -> float:
    p = point["p"]
    return float(rv_discrete(values=([1, 2, 3], p)).logpmf(2))

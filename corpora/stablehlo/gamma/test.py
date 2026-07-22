"""Independent scipy oracle: Gamma(shape, rate) logdensity at xobs."""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.gamma.logpdf(point["xobs"], a=point["shape"], scale=1 / point["rate"]))

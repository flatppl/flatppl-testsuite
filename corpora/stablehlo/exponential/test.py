"""Independent scipy oracle: Exponential(rate) logdensity at xobs."""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.expon.logpdf(point["xobs"], scale=1 / point["rate"]))

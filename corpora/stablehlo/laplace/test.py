"""Independent scipy oracle: Laplace(location, scale) logdensity at xobs."""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.laplace.logpdf(point["xobs"], loc=point["location"], scale=point["scale"]))

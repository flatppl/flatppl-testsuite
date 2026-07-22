"""Independent scipy oracle: Normal(mu, sigma) logdensity at xobs."""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.norm.logpdf(point["xobs"], loc=point["mu"], scale=point["sigma"]))

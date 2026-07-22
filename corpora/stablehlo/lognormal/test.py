"""Independent scipy oracle: LogNormal(mu, sigma) logdensity at xobs."""
from scipy import stats
import math


def oracle(point: dict) -> float:
    return float(stats.lognorm.logpdf(point["xobs"], s=point["sigma"], scale=math.exp(point["mu"])))

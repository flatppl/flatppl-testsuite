"""Independent scipy oracle: Beta(alpha, beta) logdensity at xobs."""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.beta.logpdf(point["xobs"], point["alpha"], point["beta"]))

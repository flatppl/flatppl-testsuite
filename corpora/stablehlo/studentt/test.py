"""Independent scipy oracle: StudentT(nu) logdensity at xobs."""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.t.logpdf(point["xobs"], df=point["nu"]))

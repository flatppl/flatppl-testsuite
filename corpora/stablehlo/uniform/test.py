"""Independent scipy oracle: Uniform(support = interval(-1, 3)) logdensity at xobs."""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.uniform.logpdf(point["xobs"], loc=-1, scale=4))

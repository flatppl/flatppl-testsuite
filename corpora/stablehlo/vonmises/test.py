"""§08 VonMises on its declared angular support, scored by SciPy."""
import math

from scipy import stats


def oracle(point: dict) -> float:
    if not -math.pi <= point["xobs"] <= math.pi:
        return -math.inf
    return float(stats.vonmises.logpdf(
        point["xobs"], point["kappa"], loc=point["mu"],
    ))

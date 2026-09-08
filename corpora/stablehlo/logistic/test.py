"""§08 Logistic(mu, s), independently scored by SciPy.

With z=(x-mu)/s, log f = -log(s)-z-2*logaddexp(0,-z).
Both tails require finite log densities.
"""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.logistic.logpdf(point["xobs"], loc=point["mu"], scale=point["s"]))

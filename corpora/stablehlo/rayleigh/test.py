"""§08 Rayleigh(scale), independently scored by SciPy.

The entry's density is x/scale**2 * exp(-x**2/(2 scale**2)) for x >= 0. It
is 0 at x = 0, so the log-density there is -inf, as it is below 0.
"""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.rayleigh.logpdf(point["xobs"], scale=point["scale"]))

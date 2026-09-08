"""§08 Weibull(shape, scale), independently scored by SciPy weibull_min.

For x>0, log f = log(shape/scale) + (shape-1)*log(x/scale)
- (x/scale)**shape. Negative values have zero density. Shapes on either
side of 1 distinguish the near-zero behavior and the rate/scale convention.
"""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.weibull_min.logpdf(
        point["xobs"], point["shape"], scale=point["scale"],
    ))

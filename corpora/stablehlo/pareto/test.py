"""§08 Pareto(shape, scale), independently scored by SciPy.

The entry's density is shape*scale**shape/x**(shape+1) for x>=scale,
zero below scale. This pins the parameter-dependent lower boundary even
though the current catalog's summary support column is broader.
"""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.pareto.logpdf(point["xobs"], point["shape"], scale=point["scale"]))

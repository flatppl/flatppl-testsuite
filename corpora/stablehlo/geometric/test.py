"""Independent scipy oracle: Geometric(p) logdensity at the fixed observed a=3.

FlatPPL Geometric(p) counts FAILURES before a success (support {0,1,...});
scipy.stats.geom counts TRIALS (support {1,2,...}) -- geom(p, loc=-1) shifts
back to the failure-count convention (loc=-1 is required to match).
"""
from scipy.stats import geom


def oracle(point: dict) -> float:
    return float(geom.logpmf(3, point["p"], loc=-1))

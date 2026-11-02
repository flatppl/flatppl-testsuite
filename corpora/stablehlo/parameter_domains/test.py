"""Independent scipy oracle for four §08 laws whose parameters are runtime inputs.

Spec §08 declares `sigma`, `shape` over posreals, `rate` over nonnegreals and `p`
over stdsimplex(n). A parameter outside its set has no density; the device status
reports it (StableHLO status 9). A point outside the support still scores -inf.
"""
import math

from scipy import stats

ERROR = {"error": "outside its declared value set"}


def oracle(point: dict) -> float | dict:
    sigma, rate, probs, shape = point["sigma"], point["rate"], point["probs"], point["shape"]
    simplex = all(p >= 0 for p in probs) and abs(sum(probs) - 1) <= 8 * len(probs) * 2.0**-23
    if not (sigma > 0 and rate >= 0 and shape > 0 and simplex):
        return ERROR
    c = point["c"]
    categorical = math.log(probs[c - 1]) if 1 <= c <= len(probs) and probs[c - 1] > 0 else -math.inf
    return float(
        stats.norm.logpdf(point["x"], loc=0.5, scale=sigma)
        + stats.poisson.logpmf(point["k"], rate)
        + categorical
        + stats.gamma.logpdf(point["y"], a=shape, scale=0.5)
    )

"""Independent SciPy reference for Rayleigh(scale) sampling.

Spec §08 Rayleigh's scale is sigma, matching scipy rayleigh's scale. The
kurtosis is 3.245, so at n=100000 the variance estimator's relative SE
sqrt((3.245-1)/n) is 0.0047, and the 3% moment band exceeds six SEs.
"""
PARAMS = {"scale": 1.5}


def stat() -> dict:
    return {
        "distribution": {"family": "rayleigh", "kwargs": {"scale": PARAMS["scale"]}},
        "discrete": False,
    }

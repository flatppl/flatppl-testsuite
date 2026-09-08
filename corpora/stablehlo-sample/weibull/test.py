"""Independent SciPy reference for Weibull sampling.

Spec §08 uses shape k and scale lambda, matching scipy weibull_min(c=k, scale=lambda).
"""
PARAMS = {"shape":1.7,"scale":2.3}


def stat() -> dict:
    return {
        "distribution": {
            "family": "weibull_min",
            "kwargs": {"c": PARAMS["shape"], "scale": PARAMS["scale"]},
        },
        "discrete": False,
    }

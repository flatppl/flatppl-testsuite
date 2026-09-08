"""Independent SciPy reference for Logistic sampling.

SciPy logistic uses loc=mu and scale=s from spec §08.
"""
PARAMS = {"mu":0.3,"s":1.2}


def stat() -> dict:
    return {
        "distribution": {
            "family": "logistic",
            "kwargs": {"loc": PARAMS["mu"], "scale": PARAMS["s"]},
        },
        "discrete": False,
    }

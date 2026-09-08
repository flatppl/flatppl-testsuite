"""Independent SciPy reference for InverseGamma sampling.

Spec §08 scale is the reciprocal Gamma's rate, matching scipy invgamma
scale directly (not its reciprocal). Shape 6 has finite fourth moment and
kurtosis 22. At n=400000 the variance estimator's relative SE
sqrt((22-1)/n) is 0.00725, so the suite's unchanged 5% variance band exceeds
six asymptotic SEs. The draw count is selected before engine execution.
"""
PARAMS = {"shape":6,"scale":2.5}


def stat() -> dict:
    return {
        "distribution": {
            "family": "invgamma",
            "kwargs": {"a": PARAMS["shape"], "scale": PARAMS["scale"]},
        },
        "discrete": False,
    }

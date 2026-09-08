"""Independent SciPy reference for Pareto sampling.

Spec §08 uses tail index shape and minimum scale, matching scipy pareto.
Shape 8 has finite fourth moment and kurtosis 22.725. At n=400000 the
variance estimator's relative SE sqrt((22.725-1)/n) is 0.00737, so the
suite's unchanged 5% variance band exceeds six asymptotic SEs. The draw
count is selected analytically, independently of engine results.
"""
PARAMS = {"shape":8,"scale":1.3}


def stat() -> dict:
    return {
        "distribution": {
            "family": "pareto",
            "kwargs": {"b": PARAMS["shape"], "scale": PARAMS["scale"]},
        },
        "discrete": False,
    }

"""Independent scipy reference: Laplace(location, scale) sample checks.

`stat()` is the sample test_type's contract counterpart to logdensity's
`oracle(point)` — it runs ONLY under `regen.py`, which freezes its return
value into test.json's `stat`. `distribution` is a `{"family", "kwargs"}`
recipe (not a bare number): `sample_checks.py` reconstructs a live scipy
frozen distribution from it at test time for the KS test's `.cdf` — a
deterministic scipy lookup, not new oracle computation.
"""
PARAMS = {"location": 0.0, "scale": 1.0}


def stat() -> dict:
    return {
        "distribution": {
            "family": "laplace",
            "kwargs": {"loc": PARAMS["location"], "scale": PARAMS["scale"]},
        },
        "discrete": False,
    }

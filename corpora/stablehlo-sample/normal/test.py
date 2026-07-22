"""Independent scipy reference: Normal(mu, sigma) sample checks.

`stat()` is the sample test_type's contract counterpart to logdensity's
`oracle(point)` — it runs ONLY under `regen.py`, which freezes its return
value into test.json's `stat`. `distribution` is a `{"family", "kwargs"}`
recipe (not a bare number): `sample_checks.py` reconstructs a live scipy
frozen distribution from it at test time for the KS test's `.cdf` — a
deterministic scipy lookup, not new oracle computation.
"""
PARAMS = {"mu": 0.3, "sigma": 1.2}


def stat() -> dict:
    return {
        "distribution": {"family": "norm", "kwargs": {"loc": PARAMS["mu"], "scale": PARAMS["sigma"]}},
        "discrete": False,
    }

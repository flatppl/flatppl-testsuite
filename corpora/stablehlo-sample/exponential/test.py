"""Independent scipy reference: Exponential(rate) sample checks.

`stat()` is the sample test_type's contract counterpart to logdensity's
`oracle(point)` — it runs ONLY under `regen.py`, which freezes its return
value into test.json's `stat`. `distribution` is a `{"family", "kwargs"}`
recipe (not a bare number): `sample_checks.py` reconstructs a live scipy
frozen distribution from it at test time for the KS test's `.cdf` — a
deterministic scipy lookup, not new oracle computation.
"""
PARAMS = {"rate": 1.5}


def stat() -> dict:
    return {
        "distribution": {"family": "expon", "kwargs": {"scale": 1.0 / PARAMS["rate"]}},
        "discrete": False,
    }

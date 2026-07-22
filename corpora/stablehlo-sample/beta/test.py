"""Independent scipy reference: Beta(alpha, beta) sample checks.

`stat()` is the sample test_type's contract counterpart to logdensity's
`oracle(point)` — it runs ONLY under `regen.py`, which freezes its return
value into test.json's `stat`. `distribution` is a `{"family", "kwargs"}`
recipe (not a bare number): `sample_checks.py` reconstructs a live scipy
frozen distribution from it at test time for the KS test's `.cdf` — a
deterministic scipy lookup, not new oracle computation. `independence` marks
this as the beta subject for `check_independence`: @sample builds Beta from
X/(X+Y) with X, Y drawn off two SEPARATE internal Gamma rng streams, so a
shared-stream bug would collapse every draw to 0.5.
"""
PARAMS = {"alpha": 2.0, "beta": 3.0}


def stat() -> dict:
    return {
        "distribution": {
            "family": "beta",
            "kwargs": {"a": PARAMS["alpha"], "b": PARAMS["beta"]},
        },
        "discrete": False,
        "independence": "beta",
    }

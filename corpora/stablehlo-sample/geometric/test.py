"""Independent scipy reference: Geometric(p) sample checks.

`stat()` is the sample test_type's contract counterpart to logdensity's
`oracle(point)` — it runs ONLY under `regen.py`, which freezes its return
value into test.json's `stat`. `distribution` is a `{"family", "kwargs"}`
recipe (not a bare number): `sample_checks.py` reconstructs a live scipy
frozen distribution from it at test time for the KS test's `.cdf` — a
deterministic scipy lookup, not new oracle computation. `loc=-1` shifts
scipy's {1, 2, …} (trials-to-first-success) support to FlatPPL's {0, 1, …}
(failures-before-first-success).
"""
PARAMS = {"p": 0.3}


def stat() -> dict:
    return {
        "distribution": {
            "family": "geom",
            "kwargs": {"p": PARAMS["p"], "loc": -1},
        },
        "discrete": True,
        "fanout_discrete_kmin": 0,
    }

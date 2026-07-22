"""Independent scipy reference: Categorical0(p) sample checks.

`stat()` is the sample test_type's contract counterpart to logdensity's
`oracle(point)` — it runs ONLY under `regen.py`, which freezes its return
value into test.json's `stat`. `distribution` is a `{"family", "kwargs"}`
recipe (not a bare number): `sample_checks.py` reconstructs a live scipy
frozen distribution from it at test time for the KS test's `.cdf` — a
deterministic scipy lookup, not new oracle computation. FlatPPL's
`Categorical0` is 0-indexed ({0, …, k-1}); `scipy.stats.rv_discrete(values=
...)` reconstructs that exact support/pmf recipe.
"""
PARAMS = {"p": [0.2, 0.3, 0.5]}


def stat() -> dict:
    p = PARAMS["p"]
    k = list(range(len(p)))
    return {
        "distribution": {
            "family": "rv_discrete",
            "kwargs": {"values": [k, p]},
        },
        "discrete": True,
        "fanout_discrete_kmin": 0,
    }

"""Independent scipy reference: NegativeBinomial(alpha, beta) sample checks.

`stat()` is the sample test_type's contract counterpart to logdensity's
`oracle(point)` — it runs ONLY under `regen.py`, which freezes its return
value into test.json's `stat`. `distribution` is a `{"family", "kwargs"}`
recipe (not a bare number): `sample_checks.py` reconstructs a live scipy
frozen distribution from it at test time for the KS test's `.cdf` — a
deterministic scipy lookup, not new oracle computation. scipy's `nbinom(n,
p)` counts successes-until-`n`-failures with per-trial success prob `p`;
FlatPPL's `NegativeBinomial(alpha, beta)` is the Gamma-Poisson mixture with
shape `alpha` and rate `beta`, matched via `n=alpha`, `p=beta/(beta+1)`.
"""
PARAMS = {"alpha": 5.0, "beta": 2.0}


def stat() -> dict:
    alpha, beta = PARAMS["alpha"], PARAMS["beta"]
    return {
        "distribution": {
            "family": "nbinom",
            "kwargs": {"n": alpha, "p": beta / (beta + 1.0)},
        },
        "discrete": True,
        "fanout_discrete_kmin": 0,
    }

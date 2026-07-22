"""Independent scipy reference: NegativeBinomial2(mu, psi) sample checks.

`stat()` is the sample test_type's contract counterpart to logdensity's
`oracle(point)` — it runs ONLY under `regen.py`, which freezes its return
value into test.json's `stat`. `distribution` is a `{"family", "kwargs"}`
recipe (not a bare number): `sample_checks.py` reconstructs a live scipy
frozen distribution from it at test time for the KS test's `.cdf` — a
deterministic scipy lookup, not new oracle computation. scipy's `nbinom(n,
p)` counts successes-until-`n`-failures with per-trial success prob `p`;
FlatPPL's `NegativeBinomial2(mu, psi)` is the mean/dispersion (Stan-style)
parameterization, matched via `n=psi`, `p=psi/(mu+psi)`.
"""
PARAMS = {"mu": 3.0, "psi": 5.0}


def stat() -> dict:
    mu, psi = PARAMS["mu"], PARAMS["psi"]
    return {
        "distribution": {
            "family": "nbinom",
            "kwargs": {"n": psi, "p": psi / (mu + psi)},
        },
        "discrete": True,
        "fanout_discrete_kmin": 0,
    }

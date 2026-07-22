"""Independent scipy reference: Binomial(n, p) sample checks.

`stat()` is the sample test_type's contract counterpart to logdensity's
`oracle(point)` — it runs ONLY under `regen.py`, which freezes its return
value into test.json's `stat`. `distribution` is a `{"family", "kwargs"}`
recipe (not a bare number): `sample_checks.py` reconstructs a live scipy
frozen distribution from it at test time for the KS test's `.cdf` — a
deterministic scipy lookup, not new oracle computation. `n` is a fixed-phase
literal in model.flatppl (not an ABI input) — Binomial's static-shape
StableHLO sampler needs `n` as a Rust `u64` at emit time, not a runtime
value (see flatppl-rust's `binomial_sample_refuses_parameterized_n` golden
test); `p` alone is the ABI parameter.
"""
PARAMS = {"n": 10, "p": 0.4}


def stat() -> dict:
    return {
        "distribution": {
            "family": "binom",
            "kwargs": {"n": PARAMS["n"], "p": PARAMS["p"]},
        },
        "discrete": True,
        "fanout_discrete_kmin": 0,
    }

#!/usr/bin/env python3
"""Generate ``corpora/fragment/<name>/expected.json`` from an INDEPENDENT
scipy oracle.

This is the scipy counterpart of the frozen values that were originally
derived from Julia's ``Distributions.jl`` (see each fragment's
``reference_backend``): every fragment below is a small closed-form
probability calculation, so scipy.stats reproduces the frozen numbers to
machine precision (checked to ``<= 1e-12`` below) without needing a second
run through Julia. Regenerating from this script re-derives the SAME
independent oracle value that gen_expected already froze — it is not a
second, different oracle picked to make the gate pass.

Each fragment already ends in ``lp = logdensityof(m, <point>)`` at a FIXED
point, so there is no theta scan here (unlike the HS3 corpus) — one scalar
check per model, kind ``logdensity_value``, binding ``lp``.

Not on the default test path (``pixi run test`` does not import this module).
Run it manually to verify / regenerate:

    pixi run python corpora/fragment/gen_expected.py

``±inf`` cannot round-trip through standard JSON, so ``frag_trunc_out``'s
value is written as the STRING ``"-inf"`` (parsed back to ``float("-inf")`` by
the harness — see ``suites/fragment_gate.py::_parse_expected``).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import norm

HERE = Path(__file__).resolve().parent

# Frozen values (originally computed via Julia Distributions.jl 0.25; see
# corpora/fragment/README.md). This script's job is to independently
# reproduce them via scipy, not to define them.
FROZEN = {
    "frag_superpose": -0.60623693588018757,
    "frag_trunc_in": -1.0439385332046727,
    "frag_trunc_out": float("-inf"),
    "frag_norm_trunc": -0.84377223888021002,
    "frag_pushfwd_affine": -1.7370857137646181,
    "frag_pushfwd_exp": -1.4066046182594198,
    "frag_kchain_bern": -1.6282033113610439,
    "frag_kchain_cat": -2.1367953170065803,
}

# frag_broadcast_poisson has no Julia predecessor — scipy IS its canonical
# oracle (see its expected.json reference_backend) — so it is deliberately
# absent from FROZEN; main()'s cross-check loop skips the
# reproduce-to-1e-12 assertion for any test_id not in this dict and just
# writes the scipy value directly.


def oracle_superpose() -> float:
    """`m = superpose(Normal(0,1), Normal(1,2))`; superpose is measure
    addition (§06) — the two component DENSITIES (not a normalized mixture)
    add at a point."""
    return math.log(norm.pdf(0.5, 0, 1) + norm.pdf(0.5, 1, 2))


def oracle_trunc_in() -> float:
    """`truncate` is an UNNORMALIZED gate: inside the interval the density is
    just the parent density (no renormalization by the interval mass)."""
    return norm.logpdf(0.5, 0, 1)


def oracle_trunc_out() -> float:
    """Outside the truncation interval, the gated density is zero -> -inf."""
    return float("-inf")


def oracle_norm_trunc() -> float:
    """`normalize(truncate(...))` renormalizes by the interval's probability
    mass; this is exactly `scipy.stats.truncnorm.logpdf`."""
    return norm.logpdf(0.5, 0, 1) - math.log(norm.cdf(2, 0, 1) - norm.cdf(-1, 0, 1))


def oracle_pushfwd_affine() -> float:
    """Y = 2X for X ~ Normal(0,1): the change-of-variables log-density is
    `log f_X(x) - log|dy/dx|` at `x = y/2`; equals `Normal(0,2).logpdf(1.0)`
    directly since scaling a Gaussian's scale by 2 is exact."""
    direct = norm.logpdf(0.5, 0, 1) - math.log(2)
    cross_check = norm.logpdf(1.0, 0, 2)
    assert math.isclose(direct, cross_check, rel_tol=0, abs_tol=1e-12)
    return direct


def oracle_pushfwd_exp() -> float:
    """Y = exp(X) for X ~ Normal(0,1) is LogNormal(s=1); the pushforward
    log-density at y=1.5 is `Normal(0,1).logpdf(log y) - log y`, which is
    exactly `scipy.stats.lognorm(s=1).logpdf(1.5)`."""
    from scipy.stats import lognorm

    direct = norm.logpdf(math.log(1.5), 0, 1) - math.log(1.5)
    cross_check = lognorm.logpdf(1.5, s=1)
    assert math.isclose(direct, cross_check, rel_tol=0, abs_tol=1e-12)
    return direct


def oracle_kchain_bern() -> float:
    """`z ~ Bernoulli(p=0.3)`, `y | z ~ Normal(mu=z, sigma=1)`; the marginal
    of y at 1.5 is the Bernoulli-weighted 2-component mixture
    `(1-p) N(1.5; 0, 1) + p N(1.5; 1, 1)`."""
    p = 0.3
    return math.log((1 - p) * norm.pdf(1.5, 0, 1) + p * norm.pdf(1.5, 1, 1))


def oracle_kchain_cat() -> float:
    """`z ~ Categorical(p=[0.2,0.3,0.5])` over 1-based atoms {1,2,3},
    `y | z ~ Normal(mu=z, sigma=1)`; the marginal of y at 0.5 is the
    Categorical-weighted 3-component mixture."""
    weights = [0.2, 0.3, 0.5]
    mus = [1, 2, 3]
    return math.log(sum(w * norm.pdf(0.5, mu, 1) for w, mu in zip(weights, mus)))


def oracle_broadcast_poisson() -> float:
    """`broadcast(Poisson, [2.0, 3.5, 1.0])` is an array-of-kernels measure
    over the length-3 observation array (spec §04 broadcasting); its
    log-density at `[1, 4, 2]` is the SUM of independent per-cell Poisson
    log-pmfs, `Σᵢ Poisson.logpmf(kᵢ; λᵢ)` for λ=[2.0, 3.5, 1.0], k=[1, 4, 2]."""
    from scipy.stats import poisson

    lambdas = [2.0, 3.5, 1.0]
    ks = [1, 4, 2]
    return sum(poisson.logpmf(k, lam) for k, lam in zip(ks, lambdas))


ORACLES = {
    "frag_superpose": ("superpose", oracle_superpose, "julia Distributions.jl 0.25"),
    "frag_trunc_in": ("trunc_in", oracle_trunc_in, "julia Distributions.jl 0.25"),
    "frag_trunc_out": ("trunc_out", oracle_trunc_out, "julia Distributions.jl 0.25"),
    "frag_norm_trunc": ("norm_trunc", oracle_norm_trunc, "julia Distributions.jl 0.25"),
    "frag_pushfwd_affine": ("pushfwd_affine", oracle_pushfwd_affine, "julia Distributions.jl 0.25"),
    "frag_pushfwd_exp": ("pushfwd_exp", oracle_pushfwd_exp, "julia Distributions.jl 0.25"),
    "frag_kchain_bern": ("kchain_bern", oracle_kchain_bern, "julia Distributions.jl 0.25"),
    "frag_kchain_cat": ("kchain_cat", oracle_kchain_cat, "julia Distributions.jl 0.25"),
    "frag_broadcast_poisson": ("broadcast_poisson", oracle_broadcast_poisson, "scipy.stats.poisson"),
}


def _json_expected(value: float) -> float | str:
    """±inf has no JSON literal; freeze it as the string "-inf"/"inf"."""
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return value


def gen(test_id: str, dirname: str, value: float, reference_backend: str) -> None:
    doc = {
        "schema_version": 1,
        "test_id": test_id,
        "model": f"{dirname}.flatppl",
        "reference_backend": reference_backend,
        "checks": [
            {
                "id": "logdensity_value",
                "kind": "logdensity_value",
                "binding": "lp",
                "expected": _json_expected(value),
                "tolerance": {"atol": 1e-9, "rtol": 1e-9},
            }
        ],
    }
    out = HERE / dirname / "expected.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{test_id}: expected={value!r}")


def main() -> None:
    for test_id, (dirname, oracle_fn, reference_backend) in ORACLES.items():
        value = oracle_fn()
        frozen = FROZEN.get(test_id)
        if frozen is not None:
            if math.isinf(frozen):
                assert value == frozen, f"{test_id}: scipy={value!r} frozen={frozen!r}"
            else:
                diff = abs(value - frozen)
                assert diff <= 1e-12, (
                    f"{test_id}: scipy={value!r} frozen={frozen!r} diff={diff!r} > 1e-12"
                )
        gen(test_id, dirname, value, reference_backend)


if __name__ == "__main__":
    main()

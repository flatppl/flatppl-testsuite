#!/usr/bin/env python3
"""Generate ``corpora/bayesian_inference/<name>/expected.json`` from an
INDEPENDENT scipy oracle.

``bi1_posterior``/``bi2_posterior``/``bi3_posterior``/``bi4_posterior``
express the SAME model four different ways (an explicit ``joint`` prior, a
``lawof(record(...))`` prior, a ``disintegrate``d joint, and a
``restrict``ed joint) — so this script computes ONE closed-form value and
freezes it for all four, which is exactly the 4-way construction-equivalence
check this corpus is built to exercise. ``eight_schools_posterior`` is
Rubin's hierarchical eight-schools model, computed independently.

Every model already ends in ``lp = logdensityof(posterior, <point>)`` at a
FIXED point, so there is no theta scan here (unlike the HS3 corpus) — one
scalar check per model, kind ``logdensity_value``, binding ``lp``.

Not on the default test path (``pixi run test`` does not import this module).
Run it manually to verify / regenerate:

    pixi run python corpora/bayesian_inference/gen_expected.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import cauchy, expon, norm

HERE = Path(__file__).resolve().parent

# Frozen values (see corpora/bayesian_inference/README.md). This script's job
# is to independently reproduce them via scipy, not to define them.
FROZEN = {
    "post_bi1": -74.10185205965193,
    "post_bi2": -74.10185205965193,
    "post_bi3": -74.10185205965193,
    "post_bi4": -74.10185205965193,
    "post_eight_schools": -43.43563727714813,
}


def oracle_bi_posterior() -> float:
    """bi1-4 all express the same model: ``theta1 ~ Normal(0,1)``, ``theta2 ~
    Exponential(rate=1)``, ``a = 5*theta2``, ``b = abs(theta1)*theta2``,
    ``obs ~ iid(Normal(mu=a, sigma=b), 10)``, scored at ``theta1=0.5,
    theta2=1.0`` against the fixed 10-point ``observed_data``. A
    ``bayesupdate`` posterior's log-density at a point is the prior
    log-density plus the likelihood log-density — i.e. the joint log-density
    of (theta, data) — so bi1-4, which build that same joint four different
    ways, all freeze to this one value."""
    data = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]
    t1, t2 = 0.5, 1.0
    a = 5 * t2
    b = abs(t1) * t2
    return (
        norm.logpdf(t1, 0, 1)
        + expon.logpdf(t2, scale=1.0)
        + sum(norm.logpdf(x, a, b) for x in data)
    )


def oracle_eight_schools() -> float:
    """Rubin's eight-schools model: ``mu ~ Normal(0,5)``, ``tau ~
    half-Cauchy(0,5)`` (``normalize(truncate(Cauchy(0,5), interval(0,
    inf)))`` doubles the Cauchy density on its positive half-line), ``theta ~
    iid(Normal(mu, tau), 8)``, ``y | theta ~ Normal(theta, std_errs)``, scored
    at ``mu=0.0, tau=1.0, theta=0^8`` against the fixed
    ``y_data``/``std_errs_data``."""
    y = [28, 8, -3, 7, -1, 1, 18, 12]
    se = [15, 10, 16, 11, 9, 11, 10, 18]
    mu, tau = 0.0, 1.0
    theta = [0.0] * 8
    return (
        norm.logpdf(mu, 0, 5)
        + (math.log(2) + cauchy.logpdf(tau, 0, 5))
        + sum(norm.logpdf(th, mu, tau) for th in theta)
        + sum(norm.logpdf(yi, th, sei) for yi, th, sei in zip(y, theta, se))
    )


ORACLES = {
    "post_bi1": ("bi1_posterior", oracle_bi_posterior),
    "post_bi2": ("bi2_posterior", oracle_bi_posterior),
    "post_bi3": ("bi3_posterior", oracle_bi_posterior),
    "post_bi4": ("bi4_posterior", oracle_bi_posterior),
    "post_eight_schools": ("eight_schools_posterior", oracle_eight_schools),
}


def gen(test_id: str, dirname: str, value: float) -> None:
    doc = {
        "schema_version": 1,
        "test_id": test_id,
        "model": f"{dirname}.flatppl",
        "reference_backend": "scipy 1.18",
        "checks": [
            {
                "id": "logdensity_value",
                "kind": "logdensity_value",
                "binding": "lp",
                "expected": value,
                "tolerance": {"atol": 1e-9, "rtol": 1e-9},
            }
        ],
    }
    out = HERE / dirname / "expected.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{test_id}: expected={value!r}")


def main() -> None:
    for test_id, (dirname, oracle_fn) in ORACLES.items():
        value = oracle_fn()
        frozen = FROZEN[test_id]
        diff = abs(value - frozen)
        assert diff <= 1e-12, (
            f"{test_id}: scipy={value!r} frozen={frozen!r} diff={diff!r} > 1e-12"
        )
        gen(test_id, dirname, value)


if __name__ == "__main__":
    main()

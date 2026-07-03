#!/usr/bin/env python3
"""Generate ``corpora/sample/hier_normal/expected.json`` from the
INDEPENDENT closed-form oracle in ``corpora/sample/oracle.py``.

Unlike ``corpora/fragment/`` (each fragment ends in a fixed-point
``lp = logdensityof(m, <point>)``, one scalar to freeze), the sample corpus's
model ends in ``rand(rng, lawof(record(...)))`` — a single seed gives ONE
random realization, not a reproducible scalar. So this gate seed-sweeps N
realizations (``suites/sample_gate.py`` + ``scoring/sample_sweep.cjs``) and
compares their EMPIRICAL mean/var/cov to the model's STRUCTURAL closed-form
moments (which don't depend on N or on any particular seed), within a
Monte-Carlo tolerance sized for the sweep this gate actually runs
(N=4000, frozen below as ``N_SAMPLES``).

Tolerance = ``k * SE`` with ``k=5`` (a generous multiple of the Monte-Carlo
standard error — loose enough to keep the gate robust to ordinary sampling
noise, but nowhere near loose enough to hide a REAL miss; see the
``cov_y1_y2`` check's note for exactly how many SEs away a wrong answer
would land). SE formulas (standard asymptotic Monte-Carlo error for N iid
draws):

    SE(mean X)      = sqrt(Var(X) / N)
    SE(var X)       = Var(X) * sqrt(2 / N)                        (approx.)
    SE(cov(Y1, Y2)) = sqrt((Var(Y1)*Var(Y2) + Cov(Y1,Y2)**2) / N)  (approx.)

Not on the default test path (``pixi run test`` does not import this
module). Run it manually to regenerate:

    pixi run python corpora/sample/gen_expected.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from oracle import moments

HERE = Path(__file__).resolve().parent
N_SAMPLES = 4000
SEED_BASE = 0
K = 5  # tolerance = K * Monte-Carlo SE


def _tol(se: float) -> float:
    return K * se


def gen() -> None:
    m = moments()
    var_mu = m["var_mu"]
    var_y1, var_y2 = m["var_y1"], m["var_y2"]
    assert var_y1 == var_y2
    var_y = var_y1
    cov_y1_y2 = m["cov_y1_y2"]

    se_mean_mu = math.sqrt(var_mu / N_SAMPLES)
    se_mean_y = math.sqrt(var_y / N_SAMPLES)
    se_var_mu = var_mu * math.sqrt(2 / N_SAMPLES)
    se_var_y = var_y * math.sqrt(2 / N_SAMPLES)
    se_cov = math.sqrt((var_y1 * var_y2 + cov_y1_y2 ** 2) / N_SAMPLES)
    sigmas_from_zero = cov_y1_y2 / se_cov  # how far cov=0 (independent y1,y2) would be

    doc = {
        "schema_version": 1,
        "test_id": "sample_hier_normal",
        "model": "hier_normal.flatppl",
        "density_model": "hier_normal_density.flatppl",
        "reference_backend": "closed-form (independent oracle; see corpora/sample/oracle.py)",
        "n_samples": N_SAMPLES,
        "seed_base": SEED_BASE,
        "bindings": ["mu", "y1", "y2"],
        "checks": [
            {"id": "mean_mu", "kind": "sample_stats", "stat": "mean", "field": "mu",
             "expected": m["mean_mu"], "atol": _tol(se_mean_mu)},
            {"id": "mean_y1", "kind": "sample_stats", "stat": "mean", "field": "y1",
             "expected": m["mean_y1"], "atol": _tol(se_mean_y)},
            {"id": "mean_y2", "kind": "sample_stats", "stat": "mean", "field": "y2",
             "expected": m["mean_y2"], "atol": _tol(se_mean_y)},
            {"id": "var_mu", "kind": "sample_stats", "stat": "var", "field": "mu",
             "expected": var_mu, "atol": _tol(se_var_mu)},
            {"id": "var_y1", "kind": "sample_stats", "stat": "var", "field": "y1",
             "expected": var_y1, "atol": _tol(se_var_y)},
            {"id": "var_y2", "kind": "sample_stats", "stat": "var", "field": "y2",
             "expected": var_y2, "atol": _tol(se_var_y)},
            {
                "id": "cov_y1_y2", "kind": "sample_stats", "stat": "cov",
                "fields": ["y1", "y2"], "expected": cov_y1_y2, "atol": _tol(se_cov),
                "note": (
                    "THE SHARED-ANCESTOR CATCH: y1 and y2 share the SAME mu "
                    "draw, so Cov(y1, y2) = Var(mu) = 100. If the determinizer "
                    "had instead sampled mu independently per consumer (the "
                    "bug this whole vertical exists to rule out), y1 and y2 "
                    "would come out independent and Cov(y1, y2) would land "
                    f"near 0 -- about {sigmas_from_zero:.1f} SE away from this "
                    "tolerance band's center, i.e. unmissable. Passing this "
                    "check is the statistical proof that the sample path "
                    "preserves shared-ancestor identity end-to-end."
                ),
            },
            {
                "id": "density_consistency", "kind": "density_consistency",
                "n_points": 5, "atol": 1e-6, "rtol": 1e-6,
                "note": (
                    "For the first n_points swept realizations, the "
                    "closed-form joint log-density at that exact (mu, y1, y2) "
                    "must match logdensityof(lawof(record(mu=mu, y1=y1, "
                    "y2=y2)), <point>) scored via the det-js path against "
                    "density_model -- the sampling<->density agreement check."
                ),
            },
        ],
    }
    out = HERE / "hier_normal" / "expected.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"  cov_y1_y2: expected={cov_y1_y2}, atol={_tol(se_cov):.4f} "
          f"(independent-y1y2 would be ~{sigmas_from_zero:.1f} SE away)")


if __name__ == "__main__":
    gen()

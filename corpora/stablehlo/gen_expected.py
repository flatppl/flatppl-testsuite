#!/usr/bin/env python3
"""Freeze the StableHLO gate's expected values from the INDEPENDENT scipy
oracle (``oracle.py``), and (re)write the corpus files:

* ``<key>/<key>.flatppl`` — the FlatPPL model the emitter is fed;
* ``<key>/expected.json`` — the frozen log-density VALUE at the pinned variate
  and, for continuous parameters, the central finite-difference GRADIENT (the
  reference for Enzyme's ``jax.grad``);
* ``manifest.json`` — the machine- and human-readable fixture catalogue: for
  every fixture the exact ``(distribution, parameter values, variate)`` plus
  the scipy call used, so a second, lineage-independent oracle (Julia
  ``Distributions.jl``) can reproduce each frozen value.

Freezing is scipy's; this script never runs the FlatPPL engine or the
StableHLO executor. Run it (in any env with scipy) to regenerate:

    pixi run -e stablehlo python corpora/stablehlo/gen_expected.py
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    from . import oracle  # when imported as a package
except ImportError:  # when run as a script
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import oracle  # type: ignore

HERE = Path(__file__).resolve().parent


def gen() -> None:
    manifest_fixtures = []
    for fx in oracle.FIXTURES:
        d = HERE / fx.key
        d.mkdir(exist_ok=True)
        (d / f"{fx.key}.flatppl").write_text(fx.flatppl)
        if fx.sample_flatppl:
            (d / f"{fx.key}.sample.flatppl").write_text(fx.sample_flatppl)
        if fx.fanout_flatppl:
            (d / f"{fx.key}.iid.sample.flatppl").write_text(fx.fanout_flatppl)

        val = oracle.value(fx)
        grad = oracle.fd_gradient(fx) if fx.grad_params else {}

        expected = {
            "schema_version": 1,
            "key": fx.key,
            "distribution": fx.distribution,
            "model": f"{fx.key}.flatppl",
            "reference_backend": "scipy",
            "scipy_note": fx.scipy_note,
            "params": fx.params,
            "variate": fx.variate,
            "logdensity_value": val,
            "gradient": grad,  # {} when no continuous params
            "grad_params": list(fx.grad_params),
            "tolerance": {"value_atol": 1e-4, "grad_atol": 1e-3},
        }
        (d / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")

        manifest_fixtures.append({
            "key": fx.key,
            "distribution": fx.distribution,
            "params": fx.params,
            "variate": fx.variate_repr,
            "scipy_oracle": fx.scipy_note,
            "modes": ["logdensity"]
            + (["sample"] if fx.sample_ref or fx.independence or fx.fanout_flatppl else []),
            "grad_params": list(fx.grad_params),
            "sample_distributional": bool(fx.sample_ref),
            "sample_discrete": fx.sample_discrete,
            "independence": fx.independence,
            "fanout_n": fx.fanout_n or None,
            "fanout_dim": fx.fanout_dim or None,
            "notes": fx.notes,
        })
        print(f"{fx.key}: value={val!r}"
              + (f" grad={grad}" if grad else " (no continuous params)"))

    manifest = {
        "schema_version": 1,
        "description": (
            "StableHLO numeric-EXECUTION gate: emitted StableHLO run under "
            "Enzyme-JAX, checked as numbers + gradients vs the scipy oracle. "
            "Each fixture documents (distribution, params, variate) for a "
            "lineage-independent Julia Distributions.jl cross-check."
        ),
        "tolerances": {
            "value_atol_f32": 1e-4,
            "gradient_atol": 1e-3,
            "ks_stat_max": 0.02,
            "moment_rel_tol": 0.03,
            "independence_abs_corr_max": 0.05,
        },
        "sample_N": 100_000,
        "fixtures": manifest_fixtures,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote manifest.json ({len(manifest_fixtures)} fixtures)")


if __name__ == "__main__":
    gen()

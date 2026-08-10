"""Runner: test_type=gradient, engine=stablehlo.

Concatenate model.flatppl + query.flatppl into one module (identical to the
sibling `stablehlo/<dir>` logdensity test — gradient just also checks the
derivative of the SAME emitted `@logdensity`), emit StableHLO (`flatppl
stablehlo --mode logdensity`), and compare `stablehlo_exec.gradient`'s output
at each authored point to the frozen ANALYTIC `expected_grad` within
`grad_atol`. `grad_params` (a subset of `inputs`) names which ABI arguments to
differentiate w.r.t.; their positions in `inputs` become `argnums`. A
`grad_param` may itself be vector-valued (e.g. dirichlet's `alpha`) — `got`
and the frozen value are then both lists, compared elementwise via
`np.atleast_1d` (same convention the old stablehlo gate's `check_gradient`
used).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from flatppl_testsuite.scoring.result import CheckResult, NUMERIC_MISMATCH, UNSCOREABLE
from flatppl_testsuite.unified import stablehlo_exec as ex
from flatppl_testsuite.unified.loader import TestSpec


def run(spec: TestSpec, dir: Path) -> list[CheckResult]:
    tid = dir.name
    body = spec.body
    inputs: list[str] = body["inputs"]
    grad_params: list[str] = body["grad_params"]
    points: list[dict] = body["points"]
    expected_grad: list[dict] = body["expected_grad"]
    atol = body.get("tolerance", {}).get("grad_atol", 1e-3)

    if len(expected_grad) != len(points):
        return [CheckResult(tid, "gradient", "failed", UNSCOREABLE,
                            f"{len(points)} points but {len(expected_grad)} expected_grad "
                            "(run regen)")]

    argnums = [inputs.index(p) for p in grad_params]

    try:
        src = ex.emit_concat(dir, "logdensity")
    except ex.EmitRefused as e:
        return [CheckResult(tid, "gradient", "failed", UNSCOREABLE, f"emit refused: {e}")]

    results: list[CheckResult] = []
    for i, (pt, want) in enumerate(zip(points, expected_grad)):
        arg_values = [pt[name] for name in inputs]  # ABI order
        try:
            got = ex.gradient(src, arg_values, argnums)
        except Exception as e:
            results.append(CheckResult(
                tid, f"gradient[{i}]", "failed", UNSCOREABLE,
                f"point {pt}: Enzyme could not differentiate: {e}",
            ))
            continue
        for param, g in zip(grad_params, got):
            w = want[param]
            gv = np.atleast_1d(np.asarray(g, dtype=float))
            wv = np.atleast_1d(np.asarray(w, dtype=float))
            worst = float(np.max(np.abs(gv - wv))) if gv.shape == wv.shape else float("inf")
            ok = gv.shape == wv.shape and worst < atol
            results.append(CheckResult(
                tid, f"gradient[{i}].{param}",
                "passed" if ok else "failed",
                "" if ok else NUMERIC_MISMATCH,
                "" if ok else f"point {pt}: got {g!r}, want {w!r} (worst |Δ|={worst:.3g}, atol {atol})",
            ))
    return results

"""Runner: test_type=gradient, engine=stablehlo.

Concatenate model.flatppl + query.flatppl into one module (identical to the
sibling `stablehlo/<dir>` logdensity test — gradient just also checks the
derivative of the SAME emitted `@logdensity`), emit StableHLO (`flatppl
stablehlo --mode logdensity`), and compare `stablehlo_exec.gradient`'s output
at each authored point to the frozen ANALYTIC `expected_grad` within
`grad_atol`. `grad_params` (a subset of `inputs`) names which ABI arguments to
differentiate w.r.t.; their positions in `inputs` become `argnums`.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from flatppl_testsuite.scoring.result import CheckResult, NUMERIC_MISMATCH, UNSCOREABLE
from flatppl_testsuite.unified import stablehlo_exec as ex
from flatppl_testsuite.unified.loader import TestSpec


def _concat(dir: Path) -> str:
    model = (dir / "model.flatppl").read_text()
    query = (dir / "query.flatppl").read_text()
    return model.rstrip() + "\n" + query.lstrip()


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

    src_text = _concat(dir)
    with tempfile.NamedTemporaryFile("w", suffix=".flatppl", delete=False) as f:
        f.write(src_text)
        tmp = Path(f.name)
    try:
        src = ex.emit(tmp, "logdensity")
    except ex.EmitRefused as e:
        return [CheckResult(tid, "gradient", "failed", UNSCOREABLE, f"emit refused: {e}")]
    finally:
        tmp.unlink(missing_ok=True)

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
            ok = abs(g - w) < atol
            results.append(CheckResult(
                tid, f"gradient[{i}].{param}",
                "passed" if ok else "failed",
                "" if ok else NUMERIC_MISMATCH,
                "" if ok else f"point {pt}: got {g!r}, want {w!r} (atol {atol})",
            ))
    return results

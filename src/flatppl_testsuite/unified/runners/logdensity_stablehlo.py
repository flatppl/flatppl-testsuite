"""Runner: test_type=logdensity, engine=stablehlo.

Concatenate model.flatppl + query.flatppl into one module, emit StableHLO
(`flatppl stablehlo --mode logdensity`), execute at each authored point, and
compare to the frozen `expected` within tolerance. The query's ABI `inputs`
give the argument ORDER; a point maps name->value, reordered to that ABI order.
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
    points: list[dict] = body["points"]
    expected: list[float] = body["expected"]
    atol = body.get("tolerance", {}).get("value_atol_f32", 1e-4)

    if len(expected) != len(points):
        return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE,
                            f"{len(points)} points but {len(expected)} expected values "
                            "(run regen)")]

    src_text = _concat(dir)
    with tempfile.NamedTemporaryFile("w", suffix=".flatppl", delete=False) as f:
        f.write(src_text)
        tmp = Path(f.name)
    try:
        src = ex.emit(tmp, "logdensity")
    except ex.EmitRefused as e:
        return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE, f"emit refused: {e}")]
    finally:
        tmp.unlink(missing_ok=True)

    results: list[CheckResult] = []
    for i, (pt, want) in enumerate(zip(points, expected)):
        arg_values = [pt[name] for name in inputs]  # ABI order
        got = ex.value(src, arg_values)
        ok = abs(got - want) < atol
        results.append(CheckResult(
            tid, f"logdensity[{i}]",
            "passed" if ok else "failed",
            "" if ok else NUMERIC_MISMATCH,
            "" if ok else f"point {pt}: got {got!r}, want {want!r} (atol {atol})",
        ))
    return results

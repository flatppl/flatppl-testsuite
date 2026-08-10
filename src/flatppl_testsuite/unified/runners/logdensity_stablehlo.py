"""Runner: test_type=logdensity, engine=stablehlo.

Concatenate model.flatppl + query.flatppl into one module, emit StableHLO
(`flatppl stablehlo --mode logdensity`), execute at each authored point, and
compare to the frozen `expected` within tolerance (`|got - want| <= atol +
rtol * |want|`, reading `value_atol_f32`/`value_rtol_f32` off the test's
`tolerance` dict; `value_rtol_f32` defaults to 0.0 so a dir that doesn't set
it keeps its old absolute-only behaviour). The query's ABI `inputs` give the
argument ORDER; a point maps name->value, reordered to that ABI order.

A frozen `expected` entry may be the STRING "inf"/"-inf"/"nan" rather than a
number, because those do not round-trip through JSON and `regen._json_safe`
writes them that way for every corpus. Each entry therefore goes through
`detjs_exec.parse_expected`, which exists as the single documented reader for
that convention -- a second bare `float()` here is how the two would drift.
`corpora/stablehlo/pushfwd_asinh` and `pushfwd_log_exponential` need it: past a
map's f32 overflow the correctly rounded density IS `-inf`.
"""
from __future__ import annotations

from pathlib import Path

from flatppl_testsuite.scoring.compare import compare_scalar
from flatppl_testsuite.scoring.result import CheckResult, NUMERIC_MISMATCH, UNSCOREABLE
from flatppl_testsuite.unified import stablehlo_exec as ex
from flatppl_testsuite.unified.detjs_exec import parse_expected
from flatppl_testsuite.unified.loader import TestSpec


def run(spec: TestSpec, dir: Path) -> list[CheckResult]:
    tid = dir.name
    body = spec.body
    inputs: list[str] = body["inputs"]
    points: list[dict] = body["points"]
    expected: list[float] = body["expected"]
    tol = body.get("tolerance", {})
    atol = tol.get("value_atol_f32", 1e-4)
    rtol = tol.get("value_rtol_f32", 0.0)

    if len(expected) != len(points):
        return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE,
                            f"{len(points)} points but {len(expected)} expected values "
                            "(run regen)")]

    try:
        src = ex.emit_concat(dir, "logdensity")
    except ex.EmitRefused as e:
        return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE, f"emit refused: {e}")]

    ld_sources = ex.load_data_bindings(dir)

    results: list[CheckResult] = []
    for i, (pt, want) in enumerate(zip(points, expected)):
        # ABI order; a load_data input not in the point expands to one tensor
        # per column of its source file (declared order).
        arg_values: list = []
        for name in inputs:
            if name in pt:
                arg_values.append(pt[name])
            else:
                arg_values.extend(ex.data_columns(ld_sources[name]))
        got = ex.value(src, arg_values)
        want = parse_expected(want)
        try:
            compare_scalar(got, want, {"atol": atol, "rtol": rtol})
            ok, detail = True, ""
        except AssertionError as e:
            ok, detail = False, str(e)
        results.append(CheckResult(
            tid, f"logdensity[{i}]",
            "passed" if ok else "failed",
            "" if ok else NUMERIC_MISMATCH,
            "" if ok else f"point {pt}: {detail}",
        ))
    return results

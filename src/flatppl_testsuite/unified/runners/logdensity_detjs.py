"""Runner: test_type=logdensity, engine=det-js.

Three point-modes, chosen per test dir:

* Mode ABI (`query.flatppl` present) -- PREFERRED. The dir carries an
  inputs/outputs ABI query, so det-js scores exactly the module the StableHLO
  runner scores: `outputs` evaluated with each declared input bound to that
  point's value. One query, both engines, one frozen oracle.
* Mode A (no `points`) -- the model is self-contained and ALREADY ends in the
  scored binding at a fixed point (`lp = logdensityof(m, <point>)`), so the
  model IS the query. Scored with `score_binding`, one scalar check. This is the
  fragment/bayesian_inference shape; those dirs have no `query.flatppl`.
* Mode B (`points`, but no `query.flatppl`) -- the theta-splice fallback: the
  engine appends `__score__ = logdensityof(<binding>, <theta record>)` per point
  and re-determinizes each time. Superseded by Mode ABI wherever a query exists;
  retained for a multi-point dir that has not been given one.

A determiniser refusal (exit 3) is a SKIP, not a failure: it means the model
uses a construct outside the determiniser's density fragment.
"""
from __future__ import annotations

from pathlib import Path

from flatppl_testsuite.scoring.compare import compare_scalar
from flatppl_testsuite.scoring.result import (
    CheckResult, DETERMINIZE_SKIP, NUMERIC_MISMATCH, UNSCOREABLE,
)
from flatppl_testsuite.unified import detjs_exec as ex
from flatppl_testsuite.unified.loader import TestSpec


def _close(got: float, want: float, atol: float, rtol: float) -> bool:
    """Thin adapter over the ONE shared comparator (`scoring/compare.py`), which
    owns the ±inf and NaN rules: a truncation gate scored out of support has an
    infinite expected value and no finite tolerance band, and NaN must never
    match. This runner previously carried its own copy of that logic, which had
    already drifted from the shared one on the NaN case."""
    try:
        compare_scalar(got, want, {"atol": atol, "rtol": rtol})
        return True
    except AssertionError:
        return False


def run(spec: TestSpec, dir: Path) -> list[CheckResult]:
    tid = dir.name
    body = spec.body
    model = dir / body["model"]
    binding = body["binding"]
    tol = body.get("tolerance", {})
    atol = tol.get("atol", 1e-9)
    rtol = tol.get("rtol", 1e-9)
    points = body.get("points")

    try:
        if points is None:                                  # Mode A
            got = ex.score_binding(model, binding)
            want = ex.parse_expected(body["expected"])
            ok = _close(got, want, atol, rtol)
            return [CheckResult(
                tid, "logdensity",
                "passed" if ok else "failed",
                "" if ok else NUMERIC_MISMATCH,
                "" if ok else f"got {got!r}, want {want!r} (atol {atol}, rtol {rtol})",
            )]

        expected = body["expected"]              # Mode ABI / Mode B
        if not isinstance(expected, list) or len(expected) != len(points):
            return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE,
                                f"{len(points)} points but "
                                f"{len(expected) if isinstance(expected, list) else 'scalar'} "
                                "expected values (run regen)")]

        query = dir / "query.flatppl"
        if query.exists():                                   # Mode ABI
            fields = body.get("inputs")
            if not fields:
                return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE,
                                    "query.flatppl present but test.json declares no "
                                    "`inputs` (ABI field order)")]
            scores = ex.score_abi_points(model, query, fields, points)
        else:                                                # Mode B (splice)
            scores = [ex.log_density_at(model, binding, pt) for pt in points]

        out = []
        for i, (pt, raw, got) in enumerate(zip(points, expected, scores)):
            want = ex.parse_expected(raw)
            ok = _close(got, want, atol, rtol)
            out.append(CheckResult(
                tid, f"logdensity[{i}]",
                "passed" if ok else "failed",
                "" if ok else NUMERIC_MISMATCH,
                "" if ok else f"point {pt}: got {got!r}, want {want!r} "
                              f"(atol {atol}, rtol {rtol})",
            ))
        return out
    except ex.DeterminizeRefused as e:
        return [CheckResult(tid, "logdensity", "skipped", DETERMINIZE_SKIP, str(e))]

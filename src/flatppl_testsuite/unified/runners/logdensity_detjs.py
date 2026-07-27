"""Runner: test_type=logdensity, engine=det-js.

Two point-modes, discriminated by the presence of `points`:

* Mode A (no `points`) -- the model is self-contained and ALREADY ends in the
  scored binding at a fixed point (`lp = logdensityof(m, <point>)`), so the
  model IS the query. Scored with `score_binding`, one scalar check.
* Mode B (`points` present) -- the model carries no point; for each point the
  det-js engine appends `__score__ = logdensityof(<binding>, <theta record>)`
  and scores that, one check per point.

A determiniser refusal (exit 3) is a SKIP, not a failure: it means the model
uses a construct outside the determiniser's density fragment.
"""
from __future__ import annotations

from pathlib import Path

from flatppl_testsuite.scoring.result import CheckResult, NUMERIC_MISMATCH, UNSCOREABLE
from flatppl_testsuite.unified import detjs_exec as ex
from flatppl_testsuite.unified.loader import TestSpec

DETERMINIZE_SKIP = "DETERMINIZE_SKIP"


def _close(got: float, want: float, atol: float, rtol: float) -> bool:
    if got == want:            # exact, and makes ±inf compare equal
        return True
    if got != got or want != want:   # NaN never matches
        return False
    return abs(got - want) <= atol + rtol * abs(want)


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

        expected = body["expected"]                          # Mode B
        if not isinstance(expected, list) or len(expected) != len(points):
            return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE,
                                f"{len(points)} points but "
                                f"{len(expected) if isinstance(expected, list) else 'scalar'} "
                                "expected values (run regen)")]
        out = []
        for i, (pt, raw) in enumerate(zip(points, expected)):
            got = ex.log_density_at(model, binding, pt)
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

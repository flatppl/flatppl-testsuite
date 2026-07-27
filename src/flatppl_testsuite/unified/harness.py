"""Dispatch a test directory to its (test_type, engine) runner(s)."""
from __future__ import annotations

from pathlib import Path

from flatppl_testsuite.scoring.result import CheckResult
from flatppl_testsuite.unified.loader import TestSpec, load_test
from flatppl_testsuite.unified.runners import (
    gradient_stablehlo, logdensity_detjs, logdensity_stablehlo, sample_stablehlo,
)

# (test_type, engine) -> runner.run(spec, dir) -> list[CheckResult]
_RUNNERS = {
    ("logdensity", "stablehlo"): logdensity_stablehlo.run,
    ("sample", "stablehlo"): sample_stablehlo.run,
    ("gradient", "stablehlo"): gradient_stablehlo.run,
    ("logdensity", "det-js"): logdensity_detjs.run,
}


def run_test_dir(dir: Path, engines: list[str] | None = None) -> list[CheckResult]:
    """Run every check for `dir`, or only for `engines` if given (a subset of
    the dir's `test.json` engines) -- lets a caller gate/dispatch one engine
    at a time without one engine's skip suppressing another's checks."""
    spec: TestSpec = load_test(dir)
    out: list[CheckResult] = []
    for engine in (spec.engines if engines is None else engines):
        runner = _RUNNERS.get((spec.test_type, engine))
        if runner is None:
            out.append(CheckResult(
                Path(dir).name, f"{spec.test_type}:{engine}", "skipped",
                "NO_RUNNER", f"no runner for ({spec.test_type}, {engine})",
            ))
            continue
        out.extend(runner(spec, dir))
    return out

"""Dispatch a test directory to its (test_type, engine) runner(s)."""
from __future__ import annotations

from pathlib import Path

from flatppl_testsuite.scoring.result import CheckResult
from flatppl_testsuite.unified.loader import TestSpec, load_test
from flatppl_testsuite.unified.runners import logdensity_stablehlo, sample_stablehlo

# (test_type, engine) -> runner.run(spec, dir) -> list[CheckResult]
_RUNNERS = {
    ("logdensity", "stablehlo"): logdensity_stablehlo.run,
    ("sample", "stablehlo"): sample_stablehlo.run,
}


def run_test_dir(dir: Path) -> list[CheckResult]:
    spec: TestSpec = load_test(dir)
    out: list[CheckResult] = []
    for engine in spec.engines:
        runner = _RUNNERS.get((spec.test_type, engine))
        if runner is None:
            out.append(CheckResult(
                Path(dir).name, f"{spec.test_type}:{engine}", "skipped",
                "NO_RUNNER", f"no runner for ({spec.test_type}, {engine})",
            ))
            continue
        out.extend(runner(spec, dir))
    return out

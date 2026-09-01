"""Dispatch a test directory to its (test_type, engine) runner(s)."""
from __future__ import annotations

from pathlib import Path

from dataclasses import replace

from flatppl_testsuite.scoring.result import CheckResult
from flatppl_testsuite.unified.loader import TestSpec, load_test, merged_body
from flatppl_testsuite.unified.runners import (
    convert_detjs, gradient_stablehlo, logdensity_detjs, logdensity_stablehlo,
    sample_detjs, sample_stablehlo,
)

# (test_type, engine) -> runner.run(spec, dir) -> list[CheckResult]
_RUNNERS = {
    ("logdensity", "stablehlo"): logdensity_stablehlo.run,
    ("sample", "stablehlo"): sample_stablehlo.run,
    ("gradient", "stablehlo"): gradient_stablehlo.run,
    ("logdensity", "det-js"): logdensity_detjs.run,
    ("sample", "det-js"): sample_detjs.run,
    ("convert", "det-js"): convert_detjs.run,
}


def run_test_dir(dir: Path, engines: list[str] | None = None) -> list[CheckResult]:
    """Run every check for `dir`, or only for `engines` if given (a subset of
    the dir's `test.json` engines) -- lets a caller gate/dispatch one engine
    at a time without one engine's skip suppressing another's checks."""
    spec: TestSpec = load_test(dir)
    out: list[CheckResult] = []
    for engine in (spec.engines if engines is None else engines):
        # test_type comes off the MERGED body, so an engine override may also
        # choose the runner. `corpora/sample/hier_normal` needs it: the det-js
        # case drives the sampler, while the query the StableHLO path can score
        # for that same model is the `density_consistency` check's log-density.
        body = merged_body(spec.body, engine)
        test_type = body["test_type"]
        runner = _RUNNERS.get((test_type, engine))
        if runner is None:
            out.append(CheckResult(
                Path(dir).name, f"{test_type}:{engine}", "skipped",
                "NO_RUNNER", f"no runner for ({test_type}, {engine})",
            ))
            continue
        out.extend(runner(replace(spec, test_type=test_type, body=body), dir))
    return out

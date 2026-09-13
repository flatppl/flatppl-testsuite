"""Run the shared pyhf oracle checks through StableHLO and IREE."""
from pathlib import Path

from flatppl_testsuite.scoring.result import CheckResult
from flatppl_testsuite.unified import iree_exec
from flatppl_testsuite.unified.loader import TestSpec
from flatppl_testsuite.unified.runners.convert_detjs import run_pyhf


def run(spec: TestSpec, dir: Path) -> list[CheckResult]:
    if spec.body.get("fixture_kind") != "pyhf":
        raise ValueError("IREE conversion checks currently support only pyhf workspaces")
    return run_pyhf(dir.name, dir, spec.body, iree_exec.log_density_points)

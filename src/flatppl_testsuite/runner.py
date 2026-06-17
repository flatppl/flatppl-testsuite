"""Thin top-level runner: enumerate registered suites and aggregate results."""
from __future__ import annotations

from .scoring.result import CheckResult
from .suites.base import get_suites


def run_all(selected: set[str] | None = None,
            suites: set[str] | None = None,
            oracles: tuple[str, ...] = ()) -> list[CheckResult]:
    """Run selected (or all) registered suites and return the concatenated results."""
    out: list[CheckResult] = []
    for suite in get_suites(suites):
        out.extend(suite.run(selected=selected, oracles=oracles))
    return out


def run(selected: set[str] | None = None,
        oracles: tuple[str, ...] = ()) -> list[CheckResult]:
    """Back-compat: run all suites (currently just HS3 import)."""
    return run_all(selected=selected, oracles=oracles)

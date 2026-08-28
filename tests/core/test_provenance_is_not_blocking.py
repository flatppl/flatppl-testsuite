"""Provenance must stay OFF the blocking path.

The arrangement is one line of `pytest.ini` (`addopts = -m "not provenance"`)
plus a marker on two tests, and nothing about a passing run would show if either
half were dropped -- the suite would simply start failing on metadata again on
the next engine merge. So the contract is asserted against real collection runs
rather than by re-reading the ini file.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# The two live-provenance checks. Named explicitly: a rename should break this
# test and be re-examined, not silently drop the coverage.
PROVENANCE_TESTS = (
    "tests/sweep/test_gate.py::test_the_running_binary_matches_the_table_it_is_compared_against",
    "tests/sweep/test_sampler_gate.py::test_the_running_engine_matches_the_table_it_is_compared_against",
)


def _collect(*args: str) -> list[str]:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *args],
        cwd=ROOT, capture_output=True, text=True)
    assert out.returncode in (0, 5), out.stdout + out.stderr
    return [ln for ln in out.stdout.splitlines() if "::" in ln]


@pytest.fixture(scope="module")
def default_run() -> list[str]:
    return _collect()


def test_the_default_run_collects_neither_provenance_check(default_run):
    still_blocking = [t for t in PROVENANCE_TESTS if t in default_run]
    assert not still_blocking, (
        "these run in `pixi run test`, so an engine merge will redden this repo on "
        f"metadata again: {still_blocking}")


def test_the_default_run_still_collects_the_verdict_diffs(default_run):
    """The other half: deselecting provenance must not have taken the gate with
    it. These two are the comparison that has teeth."""
    for blocking in (
            "tests/sweep/test_gate.py::test_live_sweep_matches_the_committed_table",
            "tests/sweep/test_sampler_gate.py::test_live_sweep_matches_the_committed_table"):
        assert blocking in default_run, f"{blocking} is no longer collected"


def test_the_marker_selects_exactly_the_two_provenance_checks():
    """`pixi run provenance` is `-m provenance`, so this is what it reports."""
    selected = _collect("-m", "provenance")
    assert sorted(selected) == sorted(PROVENANCE_TESTS), selected


def test_the_function_behind_the_report_is_still_gated_by_a_blocking_test(default_run):
    """`check_provenance` itself keeps a blocking unit test -- reporting a skew is
    worthless if the check cannot detect one. That test mutates copies of the
    table, so it needs no engine to move and is not marked `provenance`."""
    assert ("tests/sweep/test_sampler_gate.py::"
            "test_the_provenance_gate_rejects_a_stale_unknown_or_missing_commit"
            ) in default_run

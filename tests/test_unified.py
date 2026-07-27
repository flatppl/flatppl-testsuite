"""Unified per-test-directory harness as pytest.

Discovers every `corpora/**/` dir that has a `test.json` and runs it through
`run_test_dir`. Test cases are parametrized per (dir, engine) pair, and each
engine is gated independently -- so a dir listing more than one engine can't
have one engine's missing prerequisites silently suppress another engine's
checks. The stablehlo runners need the `stablehlo` pixi env (jax +
enzyme_ad) and a `stablehlo`-capable `flatppl` binary (`FLATPPL_BIN`); the
det-js runner needs a resolvable `flatppl` binary and `score_flatpdl.cjs`.
Both gates are scoped per engine, NOT the module, so det-js dirs (base pixi
env, no jax/enzyme) still collect+run under plain `pixi run test`.

Run the stablehlo cases for real:
    FLATPPL_BIN=/path/to/target/release/flatppl \\
        pixi run -e stablehlo python -m pytest tests/test_unified.py
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from flatppl_testsuite.unified.harness import run_test_dir
from flatppl_testsuite.unified.loader import discover_test_dirs, load_test

_CORPORA = Path(__file__).resolve().parents[1] / "corpora"
_DIRS = discover_test_dirs(_CORPORA)
_CASES = [(d, engine) for d in _DIRS for engine in load_test(d).engines]
_CASE_IDS = [f"{d.relative_to(_CORPORA)}::{engine}" for d, engine in _CASES]


# Engines whose prerequisites must be PRESENT: a missing prerequisite for one of
# these fails instead of skipping. Without it an environment fault degrades the
# whole engine to skips and the run still exits 0 -- e.g. a wrong FLATPPL_BIN in
# the stablehlo CI step turns all 111 stablehlo cases into skips and reports
# success, silently removing the entire numeric-execution gate. That is the same
# hole `assert_results_acceptable` closes for CheckResults, one level up, and it
# takes out every case rather than one directory. CI sets this for the engine each
# step exists to exercise; a developer's local run leaves it unset and still skips
# politely.
_REQUIRED_ENGINES = {
    e.strip() for e in os.environ.get("FLATPPL_REQUIRE_ENGINES", "").split(",") if e.strip()
}


def _unavailable(engine: str, reason: str) -> None:
    """Skip -- or FAIL, if this engine was declared required."""
    if engine in _REQUIRED_ENGINES:
        pytest.fail(
            f"engine {engine!r} is required (FLATPPL_REQUIRE_ENGINES="
            f"{os.environ.get('FLATPPL_REQUIRE_ENGINES')!r}) but unavailable: {reason}"
        )
    pytest.skip(reason)


def _gate_engine(engine: str) -> None:
    """Deferred import + availability check, scoped to the one engine this
    case is about to run -- so a det-js-only case (base pixi env, no
    jax/enzyme installed at all) never touches the stablehlo import, and vice
    versa."""
    if engine == "stablehlo":
        for mod in ("jax", "enzyme_ad"):
            try:
                importlib.import_module(mod)
            except ImportError:
                _unavailable(
                    engine,
                    f"unified harness stablehlo runner needs `{mod}` (the `stablehlo` env)",
                )
        from flatppl_testsuite.unified import stablehlo_exec as ex

        if not ex.binary_supports_stablehlo():
            _unavailable(
                engine,
                "FLATPPL_BIN must point at a `flatppl` built with the `stablehlo` feature",
            )
    elif engine == "det-js":
        from flatppl_testsuite.unified import detjs_exec as ex

        if not ex.engine_available():
            _unavailable(
                engine,
                "FLATPPL_BIN / score_flatpdl.cjs must both be resolvable for the det-js runner",
            )


def assert_results_acceptable(results, allow_skip: bool) -> None:
    """Every check must have PASSED. A `failed` is obviously fatal; a `skipped`
    is fatal too unless the directory opted in with `"allow_skip": true`.

    Skips were previously tolerated everywhere, which made an all-refusing run
    report green -- a determiniser regression turning the examples corpus into
    `DETERMINIZE_SKIP`s would have shipped silently. Strict by default; a
    genuine, current inability to score is declared in the test.json where a
    reviewer can see it."""
    assert results, "no results"

    failed = [r for r in results if r.status == "failed"]
    assert not failed, "failed checks:\n" + "\n".join(
        f"  {r.check_id}: {r.tag} {r.message}" for r in failed
    )

    if not allow_skip:
        skipped = [r for r in results if r.status == "skipped"]
        assert not skipped, (
            "skipped checks (set \"allow_skip\": true in this dir's test.json if "
            "this is a known, accepted gap):\n"
            + "\n".join(f"  {r.check_id}: {r.tag} {r.message}" for r in skipped)
        )


@pytest.mark.parametrize("test_dir,engine", _CASES, ids=_CASE_IDS)
def test_unified_dir(test_dir, engine):
    _gate_engine(engine)
    results = run_test_dir(test_dir, engines=[engine])
    allow_skip = bool(load_test(test_dir).body.get("allow_skip", False))
    assert_results_acceptable(results, allow_skip=allow_skip)


# The three per-dir smoke tests that used to live here (fragment/densityof_normal,
# sample/hier_normal, hs3/fixtures/rf101_basics) were removed: each duplicated the
# parametrized case for the same directory, and each called `run_test_dir` WITHOUT
# `_gate_engine`, so on a machine with no resolvable `flatppl` binary they hard-
# errored where the parametrized case correctly skips. Now that the parametrized
# case is strict about skips, they carried nothing the parametrization does not.

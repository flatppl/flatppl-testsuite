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

from pathlib import Path

import pytest

from flatppl_testsuite.unified.harness import run_test_dir
from flatppl_testsuite.unified.loader import discover_test_dirs, load_test

_CORPORA = Path(__file__).resolve().parents[1] / "corpora"
_DIRS = discover_test_dirs(_CORPORA)
_CASES = [(d, engine) for d in _DIRS for engine in load_test(d).engines]
_CASE_IDS = [f"{d.relative_to(_CORPORA)}::{engine}" for d, engine in _CASES]


def _gate_engine(engine: str) -> None:
    """Deferred import + availability check, scoped to the one engine this
    case is about to run -- so a det-js-only case (base pixi env, no
    jax/enzyme installed at all) never touches the stablehlo import, and vice
    versa."""
    if engine == "stablehlo":
        pytest.importorskip("jax", reason="unified harness stablehlo runner needs the `stablehlo` env")
        pytest.importorskip("enzyme_ad", reason="unified harness stablehlo runner needs Enzyme-JAX")
        from flatppl_testsuite.unified import stablehlo_exec as ex

        if not ex.binary_supports_stablehlo():
            pytest.skip("FLATPPL_BIN must point at a `flatppl` built with the `stablehlo` feature")
    elif engine == "det-js":
        from flatppl_testsuite.unified import detjs_exec as ex

        if not ex.engine_available():
            pytest.skip("FLATPPL_BIN / score_flatpdl.cjs must both be resolvable for the det-js runner")


@pytest.mark.parametrize("test_dir,engine", _CASES, ids=_CASE_IDS)
def test_unified_dir(test_dir, engine):
    _gate_engine(engine)
    results = run_test_dir(test_dir, engines=[engine])
    assert results, f"{test_dir}: no results"
    failed = [r for r in results if r.status == "failed"]
    assert not failed, "\n".join(f"{r.check_id}: {r.message}" for r in failed)


def test_detjs_fragment_densityof_normal():
    d = _CORPORA / "fragment" / "densityof_normal"
    results = run_test_dir(d)
    assert results, "no checks produced"
    assert all(r.status == "passed" for r in results), [
        (r.check_id, r.status, r.message) for r in results
    ]


def test_detjs_sample_hier_normal():
    d = _CORPORA / "sample" / "hier_normal"
    results = run_test_dir(d)
    assert results
    assert all(r.status == "passed" for r in results), [
        (r.check_id, r.status, r.message) for r in results
    ]


def test_convert_hs3_rf101_basics():
    d = _CORPORA / "hs3" / "fixtures" / "rf101_basics"
    results = run_test_dir(d)
    assert results
    assert all(r.status in ("passed", "skipped") for r in results), [
        (r.check_id, r.status, r.message) for r in results
    ]

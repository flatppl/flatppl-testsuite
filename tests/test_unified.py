"""Unified per-test-directory harness as pytest.

Discovers every `corpora/**/` dir that has a `test.json` and runs it through
`run_test_dir`. The stablehlo runners need the `stablehlo` pixi env (jax +
enzyme_ad) and a `stablehlo`-capable `flatppl` binary (`FLATPPL_BIN`); that
gating is scoped to the stablehlo parametrized test below, NOT the module, so
det-js dirs (base pixi env, no jax/enzyme) still collect+run under plain
`pixi run test`.

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


def _stablehlo_gate(test_dir):
    """Deferred import + availability check, scoped to dirs whose test.json
    actually requests the `stablehlo` engine — so a det-js-only dir (base
    pixi env, no jax/enzyme installed at all) never touches this import."""
    if "stablehlo" not in load_test(test_dir).engines:
        return
    pytest.importorskip("jax", reason="unified harness stablehlo runner needs the `stablehlo` env")
    pytest.importorskip("enzyme_ad", reason="unified harness stablehlo runner needs Enzyme-JAX")
    from flatppl_testsuite.unified import stablehlo_exec as ex

    if not ex.binary_supports_stablehlo():
        pytest.skip("FLATPPL_BIN must point at a `flatppl` built with the `stablehlo` feature")


@pytest.mark.parametrize(
    "test_dir", _DIRS, ids=[str(d.relative_to(_CORPORA)) for d in _DIRS]
)
def test_unified_dir(test_dir):
    _stablehlo_gate(test_dir)
    results = run_test_dir(test_dir)
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

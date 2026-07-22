"""Unified per-test-directory harness as pytest.

Discovers every `corpora/**/` dir that has a `test.json` and runs it through
`run_test_dir`. Skips cleanly when the stablehlo executor stack (jax +
enzyme_ad, only in the `stablehlo` pixi env) or a `stablehlo`-capable `flatppl`
binary (`FLATPPL_BIN`) is absent — so base-env `pixi run test` collects+skips.

Run for real:
    FLATPPL_BIN=/path/to/target/release/flatppl \\
        pixi run -e stablehlo python -m pytest tests/test_unified.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("jax", reason="unified harness stablehlo runner needs the `stablehlo` env")
pytest.importorskip("enzyme_ad", reason="unified harness stablehlo runner needs Enzyme-JAX")

from flatppl_testsuite.unified import stablehlo_exec as ex  # noqa: E402
from flatppl_testsuite.unified.harness import run_test_dir  # noqa: E402
from flatppl_testsuite.unified.loader import discover_test_dirs  # noqa: E402

pytestmark = pytest.mark.skipif(
    not ex.binary_supports_stablehlo(),
    reason="FLATPPL_BIN must point at a `flatppl` built with the `stablehlo` feature",
)

_CORPORA = Path(__file__).resolve().parents[1] / "corpora"
_DIRS = discover_test_dirs(_CORPORA)


@pytest.mark.parametrize(
    "test_dir", _DIRS, ids=[str(d.relative_to(_CORPORA)) for d in _DIRS]
)
def test_unified_dir(test_dir):
    results = run_test_dir(test_dir)
    assert results, f"{test_dir}: no results"
    failed = [r for r in results if r.status == "failed"]
    assert not failed, "\n".join(f"{r.check_id}: {r.message}" for r in failed)

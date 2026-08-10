"""Emit-once / score-many reuse check for the load_data fixture.

`corpora/stablehlo/load_data_likelihood` exists to prove the `load_data`
signature contract (spec §13): a `load_data("data.json", cartpow(cartprod(x =
cartpow(reals, 3), y = reals), 20))` binding listed in `inputs` becomes
shape-pinned RUNTIME arguments — one tensor per column in declared order
(`tensor<20x3xf32>` for the 3-vector column x, `tensor<20xf32>` for y), the
shapes coming from the declared valueset, the values never from the file at
emit time. The harness (whatever runs the module) loads the data file and
feeds the columns. This test emits the fixture's `@logdensity` module ONCE and scores it
at two disjoint y datasets fed as the same runtime tensor arg, checking each
against the fixture's own scipy oracle (`test.py::oracle`). A module with
baked values would pass the first dataset and fail the second.

Gating mirrors tests/test_unified.py: skip without jax/enzyme_ad or a
stablehlo-capable FLATPPL_BIN, but FAIL if FLATPPL_REQUIRE_ENGINES declares
stablehlo required (so the CI step that exists to run this cannot silently
skip it).
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

from flatppl_testsuite.unified.loader import load_test_module

_DIR = Path(__file__).resolve().parents[1] / "corpora" / "stablehlo" / "load_data_likelihood"

# A second y dataset disjoint from data.json's y column, fed as the same
# runtime tensor arg. Not itself a load_data source file.
_Y2 = [4.1, 3.2, 2.5, 1.8, -1.2, 2.7, 0.4, 4.2, 5.1, 3.0,
       3.1, 3.8, 1.3, 4.3, 3.0, 2.9, 3.6, 4.4, 4.5, 3.2]


def _unavailable(reason: str) -> None:
    if "stablehlo" in os.environ.get("FLATPPL_REQUIRE_ENGINES", ""):
        pytest.fail(f"engine 'stablehlo' is required but unavailable: {reason}")
    pytest.skip(reason)


def test_load_data_module_reusable_across_datasets():
    for m in ("jax", "enzyme_ad"):
        try:
            importlib.import_module(m)
        except ImportError:
            _unavailable(f"reuse check needs `{m}` (the `stablehlo` env)")
    from flatppl_testsuite.unified import stablehlo_exec as ex

    if not ex.binary_supports_stablehlo():
        _unavailable("FLATPPL_BIN must point at a `flatppl` built with the `stablehlo` feature")

    body = json.loads((_DIR / "test.json").read_text())
    tol = body["tolerance"]
    atol, rtol = tol["value_atol_f32"], tol.get("value_rtol_f32", 0.0)
    scalars = body["points"][0]  # alpha/beta/sigma only; data comes from the file

    x_col, y_col = ex.data_columns(_DIR / "data.json")  # source order == declared order
    assert y_col != _Y2, "the second dataset must be disjoint from data.json's y column"

    src = ex.emit_concat(_DIR, "logdensity")
    oracle = load_test_module(_DIR).oracle
    scalar_args = [scalars[n] for n in ("alpha", "beta", "sigma")]
    for y in (y_col, _Y2):
        got = ex.value(src, scalar_args + [x_col, y])
        want = oracle({**scalars, "x_data": x_col, "y_data": y})
        assert abs(got - want) <= atol + rtol * abs(want), (
            f"same emitted module, y={y}: got {got}, oracle {want}"
        )

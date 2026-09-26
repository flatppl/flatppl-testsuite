"""Broadcast reductions retain each array cell and the caller's input view."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
def test_broadcast_reductions_preserve_rows_and_input_reuse(tmp_path):
    _gate_engine("stablehlo")
    (tmp_path / "model.flatppl").write_text(
        "xs = elementof(cartpow(cartpow(reals, 3), 2))\n"
        "score = sum((sum.(xs) .+ prod.(xs)) .* [2.0, 3.0])\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        "xs = elementof(cartpow(cartpow(reals, 3), 2))\n"
        'm = load_module("model.flatppl", xs = xs)\n'
        "inputs = xs\noutputs = m.score\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    weights = np.array([2.0, 3.0])
    for rows in ([[2, 3, 4], [5, 6, 7]], [[-1, -2, 4], [1, -3, 2]]):
        xs = np.array(rows, dtype=float)
        expected = ((xs.sum(axis=1) + xs.prod(axis=1)) * weights).sum()
        gradient = weights[:, None] * (1 + np.column_stack([
            xs[:, 1] * xs[:, 2], xs[:, 0] * xs[:, 2], xs[:, 0] * xs[:, 1],
        ]))
        np.testing.assert_allclose(ex.value(source, [xs]), expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(ex.gradient(source, [xs], [0]), [gradient], rtol=0, atol=1e-12)
    # The executor's product derivative currently divides by zero at zero
    # factors, including for an undotted product emitted by the baseline.
    assert ex.value(source, [[[0, -2, 4], [1, 0, 0]]]) == 7.0

"""Direct input vectors preserve selected cells, collection layers and adjoints."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("output", ["pick", "totals"])
def test_input_vector_preserves_cells_and_weighted_adjoint(tmp_path, output):
    _gate_engine("stablehlo")
    (tmp_path / "model.flatppl").write_text(
        "pick(x) = [x[4, :, :], x[1, :, :], x[4, :, :], x[2, :, :], x[1, :, :]]\n"
        "totals(x) = sum.(pick(x))\n"
        "batch_pick(x) = pick.(x)\n"
        "batch_totals(x) = totals.(x)\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        'm = load_module("model.flatppl")\nbatch_pick = m.batch_pick\nbatch_totals = m.batch_totals\n'
        "points = elementof(cartpow(cartpow(cartpow(reals, [7, 2, 3]), 3), 2))\n"
        f"inputs = points\noutputs = batch_{output}.(points)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, hlo_call = ex._jax()
    evaluate = jax.jit(lambda xs: hlo_call(xs, source=source)[0])
    rows = [3, 0, 3, 1, 0]
    points = np.arange(252, dtype=float).reshape(2, 3, 7, 2, 3) / 8 - 5
    expected_value = points[:, :, rows]
    if output == "totals":
        expected_value = expected_value.sum(axis=(-2, -1))
    weights = np.arange(1, expected_value.size + 1, dtype=float).reshape(expected_value.shape)
    np.testing.assert_array_equal(evaluate(points), expected_value)
    gradient = jax.jit(jax.grad(lambda xs: jnp.sum(evaluate(xs) * weights)))
    expected = np.zeros_like(points)
    for i, row in enumerate(rows):
        expected[:, :, row] += weights[:, :, i] if output == "pick" else weights[:, :, i, None, None]
    np.testing.assert_array_equal(gradient(points), expected)
    points[0, 0, 0, 0, 0] = np.nan
    points[0, 1, 1, 1, 2] = np.inf
    points[1, 2, 3, 0, 1] = -np.inf
    expected_value = points[:, :, rows]
    if output == "totals":
        expected_value = expected_value.sum(axis=(-2, -1))
    np.testing.assert_array_equal(evaluate(points), expected_value)


@pytest.mark.stablehlo_only
def test_input_vector_fallbacks_keep_distinct_sources(tmp_path):
    _gate_engine("stablehlo")
    (tmp_path / "model.flatppl").write_text(
        "pick(x) = [x[4], x[1], x[4], x[2], x[1]]\n"
        "mixed(x, y) = [x[4], y[1], x[4], y[2], x[1]]\n"
        "both(x, y) = [mixed(x, y), pick(x .+ 10)]\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        'm = load_module("model.flatppl")\nboth = m.both\n'
        "a = elementof(cartpow(cartpow(reals, 7), 3))\n"
        "b = elementof(cartpow(cartpow(reals, 7), 3))\n"
        "inputs = (a, b)\noutputs = both.(a, b)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    assert '"stablehlo.gather"' not in source
    jax, _, hlo_call = ex._jax()
    evaluate = jax.jit(lambda a, b: hlo_call(a, b, source=source)[0])
    a = np.arange(21, dtype=float).reshape(3, 7)
    b = -2 * a - 13
    mixed = np.stack([a[:, 3], b[:, 0], a[:, 3], b[:, 1], a[:, 0]], axis=1)
    expected = np.stack([mixed, a[:, [3, 0, 3, 1, 0]] + 10], axis=1)
    np.testing.assert_array_equal(evaluate(a, b), expected)

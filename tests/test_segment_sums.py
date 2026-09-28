"""Unequal shared-source sums preserve cell axes, duplicate rows and adjoints."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
def test_segment_sums_preserve_order_nonfinites_and_weighted_gradient(tmp_path):
    _gate_engine("stablehlo")
    selections = ([1, 3, 1], [7, 2], [4])
    parts = ", ".join(f"planes(x[{rows}, :, :])" for rows in selections)
    (tmp_path / "model.flatppl").write_text(
        'flatppl_compat = "0.1"\n'
        "planes(a) = aggregate(sum, [.k, .j], a[.i, .j, .k])\n"
        f"sums(x) = [{parts}]\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        'm = load_module("model.flatppl")\nsums = m.sums\n'
        "points = elementof(cartpow(cartpow(reals, [7, 2, 3]), 3))\n"
        "inputs = points\noutputs = sums.(points)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, hlo_call = ex._jax()
    evaluate = jax.jit(lambda xs: hlo_call(xs, source=source)[0])
    indices = [np.asarray(rows) - 1 for rows in selections]
    points = np.arange(126, dtype=float).reshape(3, 7, 2, 3) / 8 - 5
    weights = np.arange(1, 55, dtype=float).reshape(3, 3, 3, 2)
    expected = np.stack([
        points[:, rows].sum(axis=1).transpose(0, 2, 1) for rows in indices
    ], axis=1)
    np.testing.assert_array_equal(evaluate(points), expected)

    # Distinct cotangents detect swapped segment, query and feature axes.
    gradient = jax.jit(jax.grad(lambda xs: jnp.sum(evaluate(xs) * weights)))
    expected_gradient = np.zeros_like(points)
    for segment, rows in enumerate(indices):
        for row in rows:
            expected_gradient[:, row] += weights[:, segment].transpose(0, 2, 1)
    np.testing.assert_array_equal(gradient(points), expected_gradient)

    points[0, 0, 0, 0] = np.nan
    points[1, 1, 1, 2] = np.inf
    points[2, 3, 0, 1] = -np.inf
    expected = np.stack([
        points[:, rows].sum(axis=1).transpose(0, 2, 1) for rows in indices
    ], axis=1)
    np.testing.assert_array_equal(evaluate(points), expected)
    zeros = np.full_like(points, -0.0)
    actual = np.asarray(evaluate(zeros))
    np.testing.assert_array_equal(actual, np.zeros((3, 3, 3, 2)))
    assert not np.signbit(actual).any()

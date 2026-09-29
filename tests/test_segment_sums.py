"""Grouped reductions preserve cell axes, duplicate rows and adjoints."""

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


@pytest.mark.stablehlo_only
def test_computed_segment_identities_and_weighted_adjoint(tmp_path):
    _gate_engine("stablehlo")
    (tmp_path / "model.flatppl").write_text(
        "segments(x, y) = [prod([x[1], y[3], x[1]]), prod(cat([y[7]], [x[2]])),\n"
        "               sum([x[4], x[5]]), sum([x[6], x[2], x[6]])]\n"
        "mapped(x) = segments(-x, 2 * x)\nbatch(x) = mapped.(x)\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        'm = load_module("model.flatppl")\nbatch = m.batch\n'
        "points = elementof(cartpow(cartpow(cartpow(reals, 7), 3), 2))\n"
        "inputs = points\noutputs = batch.(points)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, hlo_call = ex._jax()
    evaluate = jax.jit(lambda xs: hlo_call(xs, source=source)[0])
    selections = ([0, 2, 0], [6, 1], [3, 4], [5, 1, 5])
    scales = ([-1, 2, -1], [2, -1], [-1, -1], [-1, -1, -1])
    reductions = (np.prod, np.prod, np.sum, np.sum)

    def oracle(xs):
        return np.stack([
            f(xs[..., rows] * scale, axis=-1)
            for rows, scale, f in zip(selections, scales, reductions)
        ], axis=-1)

    points = np.arange(42, dtype=float).reshape(2, 3, 7) / 8 - 3
    np.testing.assert_allclose(evaluate(points), oracle(points), rtol=1e-12, atol=1e-12)
    weights = np.arange(24, dtype=float).reshape(2, 3, 4) / 4 - 3
    gradient = jax.jit(jax.grad(lambda xs: jnp.sum(evaluate(xs) * weights)))
    expected = np.zeros_like(points)
    for output, rows in enumerate(selections):
        for lane, row in enumerate(rows):
            derivative = (
                np.prod(np.delete(points[..., rows] * scales[output], lane, axis=-1), axis=-1)
                if output < 2 else 1.0
            )
            expected[..., row] += weights[..., output] * scales[output][lane] * derivative
    np.testing.assert_allclose(gradient(points), expected, rtol=1e-12, atol=1e-12)
    # Zero-factor primals work; the executor's product adjoint is not zero-safe.
    points[0, 0, 0] = points[0, 1, 6] = 0.0
    np.testing.assert_allclose(evaluate(points), oracle(points), rtol=1e-12, atol=1e-12)
    points[0, 0, 0], points[0, 1, 6] = np.nan, np.inf
    points[1, 0, 0] = points[1, 0, 2] = 1e110
    points[1, 1, 1] = points[1, 1, 6] = 1e-200
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        np.testing.assert_allclose(evaluate(points), oracle(points), rtol=1e-12, atol=1e-12)
    zeros = np.zeros_like(points)
    actual, expected = np.asarray(evaluate(zeros)), oracle(zeros)
    np.testing.assert_array_equal(actual, expected)
    # StableHLO may omit the +0 sum initializer. Product zero signs are invariant.
    np.testing.assert_array_equal(np.signbit(actual[..., :2]), np.signbit(expected[..., :2]))

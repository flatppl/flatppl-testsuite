"""Matrix exponential values, noncommuting derivatives and matrix-cell batches."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("dtype,tolerance", [("f32", 8e-6), ("f64", 2e-10)])
@pytest.mark.parametrize("target", ["cpu", "gpu"])
def test_matexp_preserves_zero_and_noncommuting_derivatives(
    tmp_path, dtype, tolerance, target
):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "A = elementof(cartpow(reals, [2, 2]))\ninputs = A\n"
        "outputs = sum(e.matexp(A = A))\n"
    )
    source = ex.emit(query, "logdensity", dtype=dtype, target=target)
    zero = [[0.0, 0.0], [0.0, 0.0]]
    assert ex.value(source, [zero]) == pytest.approx(2, rel=tolerance)
    np.testing.assert_allclose(
        ex.gradient(source, [zero], [0]), [np.ones((2, 2))], rtol=tolerance
    )
    diagonal = [[0.0, 0.0], [0.0, 1.0]]
    assert ex.value(source, [diagonal]) == pytest.approx(1 + np.e, rel=tolerance)
    expected = [[1, np.e - 1], [np.e - 1, np.e]]
    np.testing.assert_allclose(
        ex.gradient(source, [diagonal], [0]), [expected], rtol=tolerance
    )
    # N²=0. The Frechet derivative is E + (NE+EN)/2 + NEN/6.
    # These norms exercise both sides of the 16-square common range.
    for t in (1000.0, 2.0**18, 2.0**19):
        point = [[0.0, t], [0.0, 0.0]]
        assert ex.value(source, [point]) == pytest.approx(2 + t, rel=tolerance)
        expected = [[1 + t / 2, 1], [1 + t + t * t / 6, 1 + t / 2]]
        np.testing.assert_allclose(
            ex.gradient(source, [point], [0]), [expected], rtol=tolerance
        )


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("dtype,tolerance", [("f32", 8e-6), ("f64", 2e-10)])
def test_matexp_native_batches_and_vmap_keep_matrix_cells(tmp_path, dtype, tolerance):
    _gate_engine("stablehlo")
    import jax
    import jax.numpy as jnp

    query = tmp_path / "query.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "A = elementof(cartpow(cartpow(reals, [2, 2]), 3))\ninputs = A\n"
        "outputs = e.matexp.(A)\n"
    )
    source = ex.emit(query, "logdensity", dtype=dtype)
    call = ex.call_with_status(source)
    t = np.asarray([0.0, 0.4, 8.0])
    point = np.stack(
        [np.stack([np.zeros(3), -t], axis=1), np.stack([t, np.zeros(3)], axis=1)],
        axis=1,
    )
    expected = np.stack(
        [
            np.stack([np.cos(t), -np.sin(t)], axis=1),
            np.stack([np.sin(t), np.cos(t)], axis=1),
        ],
        axis=1,
    )
    point = jnp.asarray(point, dtype=jnp.float32 if dtype == "f32" else jnp.float64)
    values = ex.expect_valid(jax.jit(call)(point))[0]
    np.testing.assert_allclose(values, expected, atol=tolerance, rtol=tolerance)
    values = ex.expect_valid(jax.jit(jax.vmap(call))(jnp.stack([point, point])))[0]
    np.testing.assert_allclose(
        values, [expected, expected], atol=tolerance, rtol=tolerance
    )

    def loss(x):
        return jnp.sum(call(x)[0][0])

    gradients = jax.jit(jax.grad(loss))(point)
    assert np.all(np.isfinite(gradients))
    np.testing.assert_allclose(gradients[0], np.ones((2, 2)), rtol=tolerance)
    for bad in (np.nan, np.inf, -np.inf):
        values = ex.expect_valid(jax.jit(call)(point.at[1, 0, 0].set(bad)))[0]
        np.testing.assert_array_equal(values[0], np.eye(2))
        assert np.all(np.isnan(values[1]))

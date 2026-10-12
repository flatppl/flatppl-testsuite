"""LKJ Cholesky draws must follow the density, not only its matrix support."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def _sampler(tmp_path, source, precision):
    query = tmp_path / "query.flatppl"
    query.write_text(source)
    module = ex.emit(query, "sample", dtype=precision, restrict_enzyme_compatible=False)
    jax, _, hlo_call = ex._jax()
    return jax.jit(lambda *args: hlo_call(*args, source=module))


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_lkj_cholesky_iid_radius_law(tmp_path, precision):
    _gate_engine("stablehlo")
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    draw = _sampler(tmp_path,
        "eta = external(posreals)\ninputs = eta\n"
        "outputs = rand(iid(LKJCholesky(4, eta), 4096), rnginit(0))\n", precision)
    key = jnp.asarray([0x123456789abcdef, 0], dtype=jnp.uint64)
    for eta in (0.5, 2.0):
        values, _ = draw(key, jnp.asarray(eta, dtype=dtype))
        assert values.devices() == {jax.devices()[0]}
        assert values.dtype == dtype
        values = np.asarray(values, dtype=np.float64)
        assert values.shape == (4096, 4, 4)
        assert np.all(np.isfinite(values))
        np.testing.assert_array_equal(np.triu(values, 1), 0)
        assert np.all(np.diagonal(values, axis1=-2, axis2=-1) > 0)
        np.testing.assert_allclose(np.square(values).sum(axis=-1), 1,
                                   atol=5e-7 if precision == "f32" else 2e-14)
        for row in range(1, 4):
            # Polar coordinates give r² ~ Beta(row/2, eta+(3-row)/2).
            radius_squared = np.square(values[:, row, :row]).sum(axis=-1)
            a, b = row / 2, eta + (3 - row) / 2
            mean = a / (a + b)
            variance = a * b / ((a + b)**2 * (a + b + 1))
            assert abs(radius_squared.mean() - mean) < 6 * np.sqrt(variance / len(values))
        # Correlations have mean zero and variance 1/(2*eta+n-1).
        correlations = values @ values.swapaxes(-1, -2)
        for i, j in zip(*np.tril_indices(4, -1)):
            rho = correlations[:, i, j]
            variance = 1 / (2 * eta + 3)
            assert abs(rho.mean()) < 6 * np.sqrt(variance / len(values))
            fourth = 3 / ((2 * eta + 3) * (2 * eta + 5))
            assert abs(np.square(rho).mean() - variance) < 6 * np.sqrt(
                (fourth - variance**2) / len(values)
            )


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_lkj_cholesky_threads_rng_and_preserves_point_mass(tmp_path, precision):
    _gate_engine("stablehlo")
    _, jnp, _ = ex._jax()
    key = jnp.asarray([0x123456789abcdef, 0], dtype=jnp.uint64)
    single = _sampler(tmp_path,
        "outputs = rand(LKJCholesky(3, 2), rnginit(0))\n", precision)
    chain = _sampler(tmp_path,
        "x, state = rand(LKJCholesky(3, 2), rnginit(0))\n"
        "y, advanced = rand(LKJCholesky(3, 2), state)\n"
        "outputs = (x, y, advanced)\n", precision)
    value, state = single(key)
    next_value, advanced = single(state)
    repeated, repeated_state = single(key)
    together = chain(key)
    np.testing.assert_array_equal(value, repeated)
    np.testing.assert_array_equal(state, repeated_state)
    np.testing.assert_array_equal(value, together[0])
    np.testing.assert_array_equal(next_value, together[1])
    np.testing.assert_array_equal(advanced, together[2])
    assert not np.array_equal(key, state)
    assert not np.array_equal(value, next_value)
    point = _sampler(tmp_path,
        "outputs = rand(iid(LKJCholesky(1, 2), 3), rnginit(0))\n", precision)
    values, unchanged = point(key)
    np.testing.assert_array_equal(values, np.ones((3, 1, 1)))
    np.testing.assert_array_equal(unchanged, key)

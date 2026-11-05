"""LKJ draws must follow the density, not only their matrix support."""

import numpy as np
import pytest
from scipy import stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def _sampler(tmp_path, source, precision):
    query = tmp_path / "query.flatppl"
    query.write_text(source)
    module = ex.emit(query, "sample", dtype=precision, restrict_enzyme_compatible=False)
    jax, _, _ = ex._jax()
    draw = jax.jit(ex.call_with_status(module))
    return lambda *args: ex.expect_valid(draw(*args))


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("family", ["LKJCholesky", "LKJ"])
def test_lkj_iid_law(tmp_path, precision, family):
    _gate_engine("stablehlo")
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    draw = _sampler(tmp_path,
        "eta = external(posreals)\ninputs = eta\n"
        f"outputs = rand(iid({family}(4, eta), 4096), rnginit(0))\n", precision)
    key = jnp.asarray([0x123456789abcdef, 0], dtype=jnp.uint64)
    for eta in (0.5, 2.0):
        values, _ = draw(key, jnp.asarray(eta, dtype=dtype))
        assert values.devices() == {jax.devices()[0]}
        assert values.dtype == dtype
        values = np.asarray(values, dtype=np.float64)
        assert values.shape == (4096, 4, 4)
        assert np.all(np.isfinite(values))
        if family == "LKJCholesky":
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
        correlations = values @ values.swapaxes(-1, -2) if family == "LKJCholesky" else values
        tolerance = 5e-7 if precision == "f32" else 2e-14
        np.testing.assert_allclose(correlations, correlations.swapaxes(-1, -2), atol=tolerance)
        np.testing.assert_allclose(np.diagonal(correlations, axis1=-2, axis2=-1), 1,
                                   atol=tolerance)
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
@pytest.mark.parametrize("family", ["LKJCholesky", "LKJ"])
def test_lkj_threads_rng_and_preserves_point_mass(tmp_path, precision, family):
    _gate_engine("stablehlo")
    _, jnp, _ = ex._jax()
    key = jnp.asarray([0x123456789abcdef, 0], dtype=jnp.uint64)
    single = _sampler(tmp_path,
        f"outputs = rand({family}(3, 2), rnginit(0))\n", precision)
    chain = _sampler(tmp_path,
        f"x, state = rand({family}(3, 2), rnginit(0))\n"
        f"y, advanced = rand({family}(3, 2), state)\n"
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
        f"outputs = rand(iid({family}(1, 2), 3), rnginit(0))\n", precision)
    values, unchanged = point(key)
    np.testing.assert_array_equal(values, np.ones((3, 1, 1)))
    np.testing.assert_array_equal(unchanged, key)


# Keys that differ only in the counter word give overlapping THREE_FRY streams.
_KEY = np.random.default_rng(20261105).integers(0, 2**63, 2).astype(np.uint64)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("n", [2, 3, 5])
def test_lkj_correlation_marginals(tmp_path, precision, n):
    """Every correlation r of LKJ(n, eta) has (r + 1) / 2 ~ Beta(b, b),
    b = eta - 1 + n / 2 (Lewandowski, Kurowicka and Joe 2009)."""
    _gate_engine("stablehlo")
    _, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    m = 20000
    draw = _sampler(tmp_path,
        "eta = external(posreals)\ninputs = eta\n"
        f"outputs = rand(iid(LKJCholesky({n}, eta), {m}), rnginit(0))\n", precision)
    pairs = list(zip(*np.tril_indices(n, -1)))
    for eta in (0.5, 1.0, 4.0):
        factor, _ = draw(jnp.asarray(_KEY), jnp.asarray(eta, dtype=dtype))
        factor = np.asarray(factor, dtype=np.float64)
        correlation = factor @ factor.swapaxes(-1, -2)
        assert np.linalg.eigvalsh(correlation).min() > 0
        b = eta - 1 + n / 2
        for i, j in pairs:
            r = correlation[:, i, j]
            p = stats.kstest((r + 1) / 2, stats.beta(b, b).cdf).pvalue
            assert p * len(pairs) > 1e-3, (eta, i, j, p)


@pytest.mark.stablehlo_only
def test_lkj_lanes_do_not_share_random_bits(tmp_path):
    """One rejection loop and one normal draw serve every row and lane. Reused
    bits would repeat matrices or correlate neighbouring draws."""
    _gate_engine("stablehlo")
    _, jnp, _ = ex._jax()
    n, m = 5, 20000
    draw = _sampler(tmp_path,
        "eta = external(posreals)\ninputs = eta\n"
        f"outputs = rand(iid(LKJCholesky({n}, eta), {m}), rnginit(0))\n", "f64")
    factor, _ = draw(jnp.asarray(_KEY), jnp.asarray(1.0))
    factor = np.asarray(factor)
    lower = np.tril_indices(n, -1)
    cells = (factor @ factor.swapaxes(-1, -2))[:, lower[0], lower[1]]
    assert len(np.unique(cells, axis=0)) == m
    a = (cells[0::2] - cells[0::2].mean(0)) / cells[0::2].std(0)
    b = (cells[1::2] - cells[1::2].mean(0)) / cells[1::2].std(0)
    cross = a.T @ b / len(a)
    assert np.abs(cross).max() < 5 / np.sqrt(len(a))

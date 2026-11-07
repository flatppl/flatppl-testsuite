"""Dirichlet draws must follow scipy's Dirichlet law, with independent lanes."""

import numpy as np
import pytest
from scipy import stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

# States come from rnginit: hand-written keys that differ only in the counter word
# give overlapping THREE_FRY streams.
SEED = 20261105
ALPHAS = {
    2: [0.3, 2.5],
    10: list(np.linspace(0.2, 3.0, 10)),
    100: list(np.random.default_rng(7).uniform(0.1, 5.0, 100)),
}


def _draws(tmp_path, d, m, alpha, precision):
    query = tmp_path / "query.flatppl"
    query.write_text(
        f"alpha = external(cartpow(posreals, {d}))\ninputs = alpha\n"
        f"outputs = rand(iid(Dirichlet(alpha = alpha), {m}), rnginit(0))\n"
    )
    module = ex.emit(query, "sample", dtype=precision, restrict_enzyme_compatible=False)
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    draw = jax.jit(ex.call_with_status(module))
    values, state = ex.expect_valid(draw(jnp.asarray(ex.rng_state(SEED)), jnp.asarray(alpha, dtype=dtype)))
    assert values.dtype == dtype
    assert not np.array_equal(np.asarray(state), ex.rng_state(SEED))
    return np.asarray(values, dtype=np.float64)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("d", [2, 10, 100])
def test_dirichlet_iid_law(tmp_path, precision, d):
    _gate_engine("stablehlo")
    alpha = np.asarray(ALPHAS[d])
    m = 4000 if d == 100 else 20000
    x = _draws(tmp_path, d, m, alpha, precision)
    assert x.shape == (m, d)
    assert np.all(x >= 0)
    np.testing.assert_allclose(x.sum(axis=1), 1, atol=1e-5 if precision == "f32" else 1e-13)

    a0 = alpha.sum()
    mean, var = stats.dirichlet.mean(alpha), stats.dirichlet.var(alpha)
    assert np.all(np.abs(x.mean(axis=0) - mean) < 5 * np.sqrt(var / m))
    # Off-diagonal covariance is -alpha_i alpha_j / (a0^2 (a0 + 1)).
    centered = x - mean
    for i, j in {(0, 1), (0, d - 1), (d // 2 - 1, d - 1)}:
        product = centered[:, i] * centered[:, j]
        oracle = -alpha[i] * alpha[j] / (a0**2 * (a0 + 1))
        assert abs(product.mean() - oracle) < 5 * product.std() / np.sqrt(m)

    # Each cell is Beta(alpha_i, a0 - alpha_i). A cell near 1 rounds to 1.0,
    # so test the complement sum there; it keeps the resolved tail.
    for i in range(d):
        rest = x[:, np.arange(d) != i].sum(axis=1)
        if np.median(x[:, i]) <= np.median(rest):
            p = stats.kstest(x[:, i], stats.beta(alpha[i], a0 - alpha[i]).cdf).pvalue
        else:
            p = stats.kstest(rest, stats.beta(a0 - alpha[i], alpha[i]).cdf).pvalue
        assert p * d > 1e-3, (i, p)


@pytest.mark.stablehlo_only
def test_dirichlet_lanes_do_not_share_random_bits(tmp_path):
    """One rejection loop serves every draw and component. Reused bits would
    repeat rows or correlate neighbouring draws."""
    _gate_engine("stablehlo")
    alpha = np.asarray(ALPHAS[10])
    x = _draws(tmp_path, 10, 20000, alpha, "f64")
    assert len(np.unique(x, axis=0)) == len(x)
    assert len(np.unique(x)) == x.size
    a = (x[0::2] - x[0::2].mean(0)) / x[0::2].std(0)
    b = (x[1::2] - x[1::2].mean(0)) / x[1::2].std(0)
    cross = a.T @ b / len(a)
    assert np.abs(cross).max() < 5 / np.sqrt(len(a))

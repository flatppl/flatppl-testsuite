"""Beta and Dirichlet draws with shapes well below one keep their small cells.

A boosted Gamma draw `g0 * U^(1/shape)` underflows float32 for a visible share of
draws at shape 0.05, and normalizing then rounds a cell to exactly 0, or gives 0/0.
A cell is exactly 0 only when its true value is below the smallest normal float
(XLA:CPU flushes subnormals to zero), so the zero share must match the Beta CDF
there.
"""

import numpy as np
import pytest
from scipy import stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

SEED = 20261107
M = 20000


def _draws(tmp_path, body, precision):
    query = tmp_path / "query.flatppl"
    query.write_text(f"outputs = rand(iid({body}, {M}), rnginit(0))\n")
    module = ex.emit(query, "sample", dtype=precision, restrict_enzyme_compatible=False)
    jax, jnp, _ = ex._jax()
    state = jnp.asarray(ex.rng_state(SEED))
    values, _ = ex.expect_valid(jax.jit(ex.call_with_status(module))(state))
    return np.asarray(values, dtype=np.float64)


def _assert_zero_share(x, law, precision):
    tiny = float(np.finfo(np.float32 if precision == "f32" else np.float64).tiny)
    p = law.cdf(tiny)
    share = np.mean(x == 0)
    assert abs(share - p) < 5 * np.sqrt(max(p, 1 / x.size) / x.size), (share, p)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_small_shape_dirichlet_cells(tmp_path, precision):
    _gate_engine("stablehlo")
    x = _draws(tmp_path, "Dirichlet([0.05, 0.05, 0.05])", precision)
    assert not np.isnan(x).any()
    np.testing.assert_allclose(x.sum(axis=1), 1, atol=1e-5 if precision == "f32" else 1e-13)
    _assert_zero_share(x, stats.beta(0.05, 0.1), precision)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_small_shape_beta_draws(tmp_path, precision):
    _gate_engine("stablehlo")
    x = _draws(tmp_path, "Beta(0.05, 0.05)", precision)
    assert not np.isnan(x).any()
    _assert_zero_share(x, stats.beta(0.05, 0.05), precision)

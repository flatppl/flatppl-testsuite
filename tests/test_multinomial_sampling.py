"""Multinomial draws follow their law, and both lowering profiles draw alike."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def _sampler(tmp_path, precision, target):
    query = tmp_path / f"query_{target}.flatppl"
    query.write_text(
        "p = external(stdsimplex(4))\ninputs = p\n"
        "outputs = rand(Multinomial(n = 50, p = p), rnginit(0))\n"
    )
    module = ex.emit(query, "sample", dtype=precision, target=target,
                     restrict_enzyme_compatible=False)
    jax, _, _ = ex._jax()
    draw = jax.jit(ex.call_with_status(module))
    return lambda *args: ex.expect_valid(draw(*args))


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_multinomial_profiles_draw_alike_and_follow_the_law(tmp_path, precision):
    """The CPU profile counts trials in a loop and the GPU profile in one
    one-hot. Both read the same uniforms, so the draws must be identical."""
    _gate_engine("stablehlo")
    _, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    p = jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=dtype)
    cpu = _sampler(tmp_path, precision, "cpu")
    gpu = _sampler(tmp_path, precision, "gpu")
    # States come from rnginit: hand-written keys that differ only in the counter word
    # give overlapping THREE_FRY streams.
    key = jnp.asarray(ex.rng_state(20261105))
    rows = []
    for _ in range(2000):
        counts, next_key = gpu(key, p)
        expected, expected_key = cpu(key, p)
        np.testing.assert_array_equal(counts, expected)
        np.testing.assert_array_equal(next_key, expected_key)
        rows.append(np.asarray(counts, dtype=np.float64))
        key = next_key
    x = np.stack(rows)
    assert np.all(x.sum(axis=1) == 50)
    q = np.asarray(p, dtype=np.float64)
    variance = 50 * q * (1 - q)
    assert np.all(np.abs(x.mean(axis=0) - 50 * q) < 5 * np.sqrt(variance / len(x)))
    # Cov(X_i, X_j) = -n p_i p_j.
    centered = x - 50 * q
    product = centered[:, 0] * centered[:, 3]
    assert abs(product.mean() + 50 * q[0] * q[3]) < 5 * product.std() / np.sqrt(len(x))

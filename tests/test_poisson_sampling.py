"""Poisson sampling must retain its law beyond the former count cutoff."""

import numpy as np
import pytest
from scipy.stats import nbinom, poisson

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def _sampler(tmp_path, source, precision):
    query = tmp_path / "query.flatppl"
    query.write_text(source)
    module = ex.emit(query, "sample", dtype=precision, restrict_enzyme_compatible=False)
    _, _, hlo_call = ex._jax()
    return lambda *args: hlo_call(*args, source=module)


def _batches(draw):
    jax, _, _ = ex._jax()

    def collect(key, *params):
        def step(state, _):
            value, advanced = draw(state, *params)
            return advanced, value

        return jax.lax.scan(step, key, None, length=16)

    return jax.jit(collect)


def _assert_count_law(values, law, mean, variance, fourth_moment):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    count = values.size
    assert np.all(np.isfinite(values)) and np.all(values >= 0)
    np.testing.assert_array_equal(values, np.floor(values))
    # Six Monte Carlo standard errors, with fixed seeds. The variance of the
    # unbiased sample variance uses the analytic fourth central moment.
    mean_se = np.sqrt(variance / count)
    variance_se = np.sqrt(
        (fourth_moment - (count - 3) / (count - 1) * variance**2) / count
    )
    assert abs(values.mean() - mean) <= 6 * mean_se + 1e-12
    assert abs(values.var(ddof=1) - variance) <= 6 * variance_se + 1e-12
    for point in (max(0, np.floor(mean - 2 * np.sqrt(variance))),
                  np.floor(mean + 2 * np.sqrt(variance))):
        expected = law.cdf(point)
        standard_error = np.sqrt(expected * (1 - expected) / count)
        assert abs(np.mean(values <= point) - expected) <= 6 * standard_error + 1 / count


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_poisson_iid_small_large_and_switch_rates(tmp_path, precision):
    _gate_engine("stablehlo")
    _, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    draw = _sampler(tmp_path,
        "rate = external(nonnegreals)\ninputs = rate\n"
        "outputs = rand(iid(Poisson(rate), 4096), rnginit(0))\n", precision)
    collect = _batches(draw)
    key = jnp.asarray([0x123456789abcdef, 0], dtype=jnp.uint64)
    for rate in (0, 4, 9.999, 10, 400, 1000, 1e6):
        _, values = collect(key, jnp.asarray(rate, dtype=dtype))
        assert values.dtype == dtype
        _assert_count_law(values, poisson(rate), rate, rate, rate + 3 * rate**2)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("measure,mean,shape", [
    ("NegativeBinomial(20, 0.05)", 400, 20),
    ("NegativeBinomial2(1000, 40)", 1000, 40),
])
def test_poisson_stage_preserves_negative_binomial_law(tmp_path, precision, measure, mean, shape):
    _gate_engine("stablehlo")
    _, jnp, _ = ex._jax()
    draw = _sampler(tmp_path,
        f"outputs = rand(iid({measure}, 4096), rnginit(0))\n", precision)
    key = jnp.asarray([0x123456789abcdef, 0], dtype=jnp.uint64)
    _, values = _batches(draw)(key)
    variance = mean + mean**2 / shape
    fourth_moment = (3 + 6 / shape) * variance**2 + variance
    law = nbinom(shape, shape / (mean + shape))
    _assert_count_law(values, law, mean, variance, fourth_moment)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_poisson_rejection_threads_complete_successor_state(tmp_path, precision):
    _gate_engine("stablehlo")
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    inputs = "rate = external(nonnegreals)\ninputs = rate\n"
    single = jax.jit(_sampler(tmp_path, inputs
        + "outputs = rand(Poisson(rate), rnginit(0))\n", precision))
    chain = jax.jit(_sampler(tmp_path, inputs
        + "x, state = rand(Poisson(rate), rnginit(0))\n"
        + "y, advanced = rand(Poisson(rate), state)\n"
        + "outputs = (x, y, advanced)\n", precision))
    key = jnp.asarray([0x123456789abcdef, 0], dtype=jnp.uint64)
    for rate in (0, 4, 10, 400, 1000):
        parameter = jnp.asarray(rate, dtype=dtype)
        value, state = single(key, parameter)
        next_value, advanced = single(state, parameter)
        repeated, repeated_state = single(key, parameter)
        together = chain(key, parameter)
        assert value.dtype == dtype
        np.testing.assert_array_equal(value, repeated)
        np.testing.assert_array_equal(state, repeated_state)
        np.testing.assert_array_equal(value, together[0])
        np.testing.assert_array_equal(next_value, together[1])
        np.testing.assert_array_equal(advanced, together[2])
        if rate > 0:
            assert not np.array_equal(state, key)

"""Conditional prior bounds retain their density and normalizer gradients."""

import math

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 2e-5), ("f64", 1e-11)])
def test_record_prior_keeps_runtime_bounds_and_normalizers(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        'reg = standard_module("regression", "0.1")\n'
        "sigma ~ Exponential(2)\n"
        "b_flat ~ iid(reg.Flat(interval(0, sigma)), 1)\n"
        "b_normal ~ iid(normalize(truncate(Normal(0, 1), interval(0, sigma))), 1)\n"
        "flat_prior = lawof(record(b = b_flat, sigma = sigma))\n"
        "normal_prior = lawof(record(b = b_normal, sigma = sigma))\n"
        "point = elementof(cartprod(b = cartpow(reals, 1), sigma = posreals))\n"
        "inputs = point\n"
        "outputs = (logdensityof(flat_prior, point), logdensityof(normal_prior, point))\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def scores(point):
        return jnp.stack(hlo_call(point[:1], point[1], source=source))

    values = jax.jit(scores)
    gradients = jax.jit(jax.jacrev(scores))
    for point in np.array([[.5, 2], [.15, .3], [2, 3]], dtype=dtype):
        b, sigma = point.astype(np.float64)
        flat_score = math.log(2) - 2 * sigma
        mass = .5 * math.erf(sigma / math.sqrt(2))
        log_phi_b = -.5 * (math.log(2 * math.pi) + b**2)
        phi_sigma = math.exp(-.5 * sigma**2) / math.sqrt(2 * math.pi)
        actual_values, actual_gradients = values(point), gradients(point)
        assert actual_values.devices() == {jax.devices()[0]}
        assert actual_gradients.devices() == {jax.devices()[0]}
        np.testing.assert_allclose(
            actual_values, [flat_score, flat_score + log_phi_b - math.log(mass)],
            rtol=tolerance, atol=tolerance,
        )
        np.testing.assert_array_equal(actual_gradients[0], [0, -2])
        np.testing.assert_allclose(
            actual_gradients[1], [-b, -2 - phi_sigma / mass],
            rtol=tolerance, atol=tolerance,
        )
    # The same compiled query must observe the new upper bound, not a captured
    # value from the earlier calls. Flat receives no finite-volume normalizer.
    outside = values(jnp.asarray([.5, .25], dtype=dtype))
    assert np.all(np.isneginf(outside))

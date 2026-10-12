"""Flat prior composition against an exact Gaussian posterior."""

import math

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 2e-5), ("f64", 1e-11)])
def test_flat_prior_scores_and_gradients_preserve_a_gaussian_posterior(
    tmp_path, precision, tolerance,
):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        'reg = standard_module("regression", "0.1")\n'
        "b ~ reg.Flat()\nz ~ Normal(0, 1)\n"
        "prior = lawof(record(b = b, z = z))\n"
        "y ~ Normal(b + z, 1)\n"
        "model = kernelof(y, b = b, z = z)\n"
        "posterior = bayesupdate(likelihoodof(model, 2), prior)\n"
        "point = elementof(cartpow(reals, 2))\n"
        "parameters = record(b = point[1], z = point[2])\n"
        "inputs = point\n"
        "outputs = (logdensityof(iid(reg.Flat(), 2), point), "
        "logdensityof(prior, parameters), logdensityof(posterior, parameters))\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def scores(point):
        return jnp.stack(hlo_call(point, source=source))

    values = jax.jit(scores)
    gradients = jax.jit(jax.jacrev(scores))
    for point in np.array([[.25, -.5], [2, 0], [-1, 1.25]], dtype=dtype):
        b, z = point.astype(np.float64)
        residual = 2 - b - z
        prior_score = -.5 * (math.log(2 * math.pi) + z**2)
        # Integrating the likelihood over b gives one. The full posterior is
        # MvNormal([2, 0], [[2, -1], [-1, 1]]), with no missing normalizer.
        posterior_score = prior_score - .5 * (math.log(2 * math.pi) + residual**2)
        actual_values, actual_gradients = values(point), gradients(point)
        assert actual_values[0] == 0
        np.testing.assert_array_equal(actual_gradients[0], [0, 0])
        np.testing.assert_allclose(
            actual_values[1:], [prior_score, posterior_score],
            rtol=tolerance, atol=tolerance,
        )
        np.testing.assert_allclose(
            actual_gradients[1:], [[0, -z], [residual, residual - z]],
            rtol=tolerance, atol=tolerance,
        )

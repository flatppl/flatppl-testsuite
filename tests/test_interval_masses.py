"""Exact truncation masses required by normalized formula priors."""

import math

import numpy as np
import pytest
from scipy import special, stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-11)])
def test_gamma_tail_and_full_support_prior_scores(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "x = external(posreals)\nrate = external(posreals)\n"
        "lower = external(reals)\ninputs = (x, rate, lower)\n"
        "outputs = ("
        + ", ".join(
            f"logdensityof(normalize(truncate(Gamma({shape}, rate), "
            "interval(lower, inf))), x)" for shape in (1, 2, 5)
        )
        + ", "
        + ", ".join(
            f"logdensityof(normalize(truncate({law}, interval(0, inf))), x) "
            f"- logdensityof({law}, x)"
            for law in ("Gamma(0.01, rate)", "InverseGamma(0.4, rate)")
        )
        + ")\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def scores(point):
        return jnp.stack(hlo_call(*point, source=source))

    values = jax.jit(scores)
    gradients = jax.jit(jax.jacrev(scores))
    for point in np.array([
        [1, 2, -1], [4, .1, 1], [13, 2, 12], [6, .1, 5],
    ], dtype=dtype):
        x, rate, lower = point.astype(np.float64)
        expected_values, expected_gradients = [], []
        for shape in (1, 2, 5):
            z = rate * max(lower, 0)
            survival = special.gammaincc(shape, z)
            log_density = (shape * math.log(rate) - special.gammaln(shape)
                           + (shape - 1) * math.log(x) - rate * x)
            expected_values.append(log_density - math.log(survival))
            hazard = (math.exp((shape - 1) * math.log(z) - z - special.gammaln(shape))
                      / survival) if lower > 0 else 0
            expected_gradients.append([
                (shape - 1) / x - rate,
                shape / rate - x + max(lower, 0) * hazard,
                rate * hazard,
            ])
        actual_values, actual_gradients = values(point), gradients(point)
        assert actual_values.devices() == {jax.devices()[0]}
        np.testing.assert_allclose(actual_values[:3], expected_values,
                                   rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(actual_gradients[:3], expected_gradients,
                                   rtol=tolerance, atol=tolerance)
        np.testing.assert_array_equal(actual_values[3:], [0, 0])
        np.testing.assert_array_equal(actual_gradients[3:], np.zeros((2, 3)))
    outside = values(jnp.asarray([.5, 1, 1], dtype=dtype))
    assert np.all(np.isneginf(outside[:3]))


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-11)])
def test_student_t_live_bounds_and_location_scale_gradients(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    degrees = (2, 4, 8, 128)
    query = tmp_path / "query.flatppl"
    query.write_text(
        "x = external(reals)\nmu = external(reals)\nsigma = external(posreals)\n"
        "lower = external(reals)\nupper = external(reals)\n"
        "inputs = (x, mu, sigma, lower, upper)\noutputs = ("
        + ", ".join(
            f"logdensityof(locscale(normalize(truncate(StudentT({nu}), "
            "interval((lower-mu)/sigma, (upper-mu)/sigma))), mu, sigma), x)"
            for nu in degrees
        )
        + ")\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def scores(point):
        return jnp.stack(hlo_call(*point, source=source))

    values = jax.jit(scores)
    gradients = jax.jit(jax.jacrev(scores))
    for point in np.array([
        [1.1, 1, 3, 0, 10], [.5, 0, 1, 0, 2], [-.5, 0, 1, -2, 0],
        [5, 0, 1, 4, 8], [-5, 0, 1, -8, -4], [.3, 0, 1, -2, 1.5],
    ], dtype=dtype):
        x, mu, sigma, lower, upper = point.astype(np.float64)
        z, a, b = (x-mu)/sigma, (lower-mu)/sigma, (upper-mu)/sigma
        expected_values, expected_gradients = [], []
        for nu in degrees:
            mass = (stats.t.sf(a, nu) - stats.t.sf(b, nu) if a >= 0
                    else stats.t.cdf(b, nu) - stats.t.cdf(a, nu))
            lower_hazard, upper_hazard = stats.t.pdf([a, b], nu) / mass
            score_derivative = -(nu + 1) * z / (nu + z*z)
            expected_values.append(stats.t.logpdf(z, nu) - math.log(sigma*mass))
            expected_gradients.append([
                score_derivative / sigma,
                (-score_derivative + upper_hazard - lower_hazard) / sigma,
                (-1 - z*score_derivative + b*upper_hazard - a*lower_hazard) / sigma,
                lower_hazard / sigma,
                -upper_hazard / sigma,
            ])
        actual_values, actual_gradients = values(point), gradients(point)
        assert actual_values.devices() == {jax.devices()[0]}
        assert actual_gradients.devices() == {jax.devices()[0]}
        np.testing.assert_allclose(actual_values, expected_values,
                                   rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(actual_gradients, expected_gradients,
                                   rtol=tolerance, atol=tolerance)

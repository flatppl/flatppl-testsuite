"""Conditional prior bounds retain their density and normalizer gradients."""

import json
import math
import subprocess

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
        "sigma ~ Exponential(2)\n"
        "reference = joint(b = iid(Lebesgue(interval(0, sigma)), 1))\n"
        "b_normal ~ iid(normalize(truncate(Normal(0, 1), interval(0, sigma))), 1)\n"
        "reference_prior = jointchain(lawof(record(sigma = sigma)), functionof(reference, sigma = sigma))\n"
        "normal_prior = lawof(record(sigma = sigma, b = b_normal))\n"
        "point = elementof(cartprod(sigma = posreals, b = cartpow(reals, 1)))\n"
        "inputs = point\n"
        "outputs = (logdensityof(reference_prior, point), logdensityof(normal_prior, point))\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def scores(point):
        return jnp.stack(hlo_call(point[1], point[:1], source=source))

    values = jax.jit(scores)
    gradients = jax.jit(jax.jacrev(scores))
    for point in np.array([[.5, 2], [.15, .3], [2, 3]], dtype=dtype):
        b, sigma = point.astype(np.float64)
        reference_score = math.log(2) - 2 * sigma
        mass = .5 * math.erf(sigma / math.sqrt(2))
        log_phi_b = -.5 * (math.log(2 * math.pi) + b**2)
        phi_sigma = math.exp(-.5 * sigma**2) / math.sqrt(2 * math.pi)
        actual_values, actual_gradients = values(point), gradients(point)
        assert actual_values.devices() == {jax.devices()[0]}
        assert actual_gradients.devices() == {jax.devices()[0]}
        np.testing.assert_allclose(
            actual_values, [reference_score, reference_score + log_phi_b - math.log(mass)],
            rtol=tolerance, atol=tolerance,
        )
        np.testing.assert_array_equal(actual_gradients[0], [0, -2])
        np.testing.assert_allclose(
            actual_gradients[1], [-b, -2 - phi_sigma / mass],
            rtol=tolerance, atol=tolerance,
        )
    # The same compiled query must observe the new upper bound, not a captured
    # value from the earlier calls. Lebesgue receives no finite-volume normalizer.
    outside = values(jnp.asarray([.5, .25], dtype=dtype))
    assert np.all(np.isneginf(outside))


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 2e-5), ("f64", 1e-11)])
@pytest.mark.parametrize("prior", ["cauchy", "student_t", "locscale"])
def test_formula_scaled_priors_preserve_domains_and_endpoint_gradients(
    tmp_path, precision, tolerance, prior,
):
    _gate_engine("stablehlo")
    # Cauchy also exercises a nonconstant truncation mass; Student-t exercises
    # the other alias. Explicit locscale retains its general reflection law.
    coefficient = {"class": "b", "prior": "student_t(4, 0, temp_Intercept)"}
    if prior == "cauchy":
        coefficient.update(prior="cauchy(0.7, temp_Intercept)", lb=0)
    elif prior == "locscale":
        coefficient.update(prior="locscale(Cauchy(0, 1), 0.7, temp_Intercept)", lb=0)
    specification = tmp_path / "model.formula.json"
    specification.write_text(json.dumps({
        "formula": "y ~ x",
        "priors": [
            {"class": "Intercept", "prior": "normal(0, 2)"},
            coefficient,
            {"class": "sigma", "prior": "constant(1)"},
        ],
        "data": {"nrows": 3, "columns": [
            {"name": "x", "values": {"kind": "real64", "values": [0, 1, 2]}},
            {"name": "y", "values": {"kind": "real64", "values": [1, 2, 3]}},
        ]},
    }))
    query = tmp_path / "query.flatppl"
    subprocess.run([str(ex.flatppl_bin()), "convert", str(specification), str(query)],
                   check=True, capture_output=True, text=True)
    query.write_text(query.read_text() + """
K_b = kernelof(b, temp_Intercept = temp_Intercept)
theta = elementof(cartpow(reals, 2))
inputs = theta
point = record(temp_Intercept = theta[1], b = theta[[2]])
outputs = (
  logdensityof(K_b(theta[1]), theta[[2]]),
  logdensityof(Pi_post, point),
  logdensityof(locscale(Cauchy(0, 1), 0, theta[1]), theta[2])
)
""")
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def scores(point):
        return jnp.stack(hlo_call(point, source=source))

    values = jax.jit(scores)
    gradients = jax.jit(jax.jacrev(scores))
    points = [[2, .4], [.3, 1.1]] + ([[-2, .4]] if prior == "locscale" else [])
    for point in np.array(points, dtype=dtype):
        scale, b = point.astype(np.float64)
        if prior != "student_t":
            delta = b - .7
            denominator = scale**2 + delta**2
            mass = .5 + math.atan(.7 / abs(scale)) / math.pi
            score = -math.log(math.pi * abs(scale)) - math.log1p((delta / scale)**2)
            score -= math.log(mass)
            derivative = np.array([
                (delta**2 - scale**2) / (scale * denominator)
                + math.copysign(.7, scale) / (math.pi * (scale**2 + .7**2) * mass),
                -2 * delta / denominator,
            ])
        else:
            score = math.log(3 / 8) - math.log(scale) - 2.5 * math.log1p(b**2 / (4 * scale**2))
            derivative = np.array([
                -1 / scale + 5 * b**2 / (scale * (4 * scale**2 + b**2)),
                -5 * b / (4 * scale**2 + b**2),
            ])
        residual = np.array([1, 2, 3]) - scale - np.array([-1, 0, 1]) * b
        posterior = score - math.log(2) - 2 * math.log(2 * math.pi)
        posterior -= scale**2 / 8 + .5 * sum(residual**2)
        posterior_gradient = derivative + [sum(residual) - scale / 4, residual[2] - residual[0]]
        actual_values, actual_gradients = values(point), gradients(point)
        assert actual_values.devices() == {jax.devices()[0]}
        assert actual_gradients.devices() == {jax.devices()[0]}
        np.testing.assert_allclose(actual_values[:2], [score, posterior],
                                   rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(actual_gradients[:2], [derivative, posterior_gradient],
                                   rtol=tolerance, atol=tolerance)
    invalid_scales = [] if prior == "locscale" else [-1, 0, np.inf, -np.inf, np.nan]
    for scale in invalid_scales:
        actual = values(jnp.asarray([scale, .4], dtype=dtype))
        assert np.all(np.isnan(actual[:2]))  # Invalid parameters are undefined.
    reflected = values(jnp.asarray([-1, .4], dtype=dtype))
    np.testing.assert_allclose(reflected[2], -math.log(math.pi) - math.log1p(.4**2),
                               rtol=tolerance, atol=tolerance)

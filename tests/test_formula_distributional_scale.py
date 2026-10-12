"""Generated positive-scale laws retain their density and sampling domains."""

import json
import math
import subprocess

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def _convert(tmp_path, specification):
    source = tmp_path / "model.formula.json"
    source.write_text(json.dumps(specification))
    model = tmp_path / "model.flatppl"
    subprocess.run([str(ex.flatppl_bin()), "convert", str(source), str(model)],
                   check=True, capture_output=True, text=True)
    return model.read_text()


def _sample(tmp_path, model, query, precision):
    path = tmp_path / "sample.flatppl"
    path.write_text(model + query)
    source = ex.emit(path, "sample", dtype=precision, restrict_enzyme_compatible=False)
    jax, _, hlo_call = ex._jax()
    return jax.jit(lambda key, point: hlo_call(key, point, source=source))


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-11)])
def test_student_identity_scale_preserves_density_and_forward_law(
    tmp_path, precision, tolerance,
):
    _gate_engine("stablehlo")
    model = _convert(tmp_path, {
        "formula": "y ~ 1", "family": "student",
        "dpars": {"sigma": {"formula": "sigma ~ 1", "link": "identity"}},
        "priors": [
            {"class": "Intercept", "prior": "normal(0, 2)"},
            {"class": "Intercept", "dpar": "sigma", "prior": "normal(0, 2)"},
            {"class": "nu", "prior": "constant(4)"},
        ],
        "data": {"nrows": 4, "columns": [
            {"name": "y", "values": {"kind": "real64", "values": [2, 4, 8, 7]}},
        ]},
    })
    inputs = "\ntheta = elementof(cartpow(reals, 2))\ninputs = theta\n"
    query = tmp_path / "score.flatppl"
    query.write_text(model + inputs + """
point = record(temp_Intercept = theta[1], temp_Intercept_sigma = theta[2])
outputs = (logdensityof(Pi_post, point), log_lik(theta[1], theta[2]))
""")
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def scores(point):
        posterior, pointwise = hlo_call(point, source=source)
        return jnp.concatenate((posterior[None], pointwise))

    values, gradients = jax.jit(scores), jax.jit(jax.jacrev(scores))
    sample = _sample(tmp_path, model, inputs + """
rstate = rnginit(19)
value, state = rand(K_model(theta[1], theta[2]), rstate)
reference, base_state = rand(K_model(theta[1], 1), rstate)
after, final_state = rand(Normal(0, 1), state)
base_after, base_final_state = rand(Normal(0, 1), base_state)
expected = theta[1] .+ (theta[2] .* (reference.y .- theta[1]))
outputs = (value.y, expected, after, base_after, final_state, base_final_state)
""", precision)
    key = jnp.asarray([19, 0], dtype=jnp.uint64)
    for point in np.asarray([[.5, .8], [.2, 1.5]], dtype=dtype):
        mu, sigma = point.astype(np.float64)
        delta = np.asarray([2, 4, 8, 7]) - mu
        denominator = 4 * sigma**2 + delta**2
        pointwise = math.log(3 / 8) - math.log(sigma) - 2.5 * np.log1p(delta**2 / (4 * sigma**2))
        derivative = np.column_stack((5 * delta / denominator,
                                      -1 / sigma + 5 * delta**2 / (sigma * denominator)))
        posterior = sum(pointwise) - 2 * math.log(2) - math.log(2 * math.pi)
        posterior -= (mu**2 + sigma**2) / 8
        posterior_gradient = derivative.sum(axis=0) - [mu / 4, sigma / 4]
        np.testing.assert_allclose(values(point), np.r_[posterior, pointwise],
                                   rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(gradients(point), np.vstack((posterior_gradient, derivative)),
                                   rtol=tolerance, atol=tolerance)
        actual = sample(key, point)
        np.testing.assert_allclose(actual[0], actual[1], rtol=tolerance, atol=tolerance)
        np.testing.assert_array_equal(actual[2], actual[3])
        np.testing.assert_array_equal(actual[4], actual[5])
    for sigma in [-1, 0, np.inf, -np.inf, np.nan]:
        point = jnp.asarray([.5, sigma], dtype=dtype)
        assert np.all(np.isnan(values(point)))
        actual = sample(key, point)
        assert np.all(np.isnan(actual[0]))
        np.testing.assert_array_equal(actual[2], actual[3])
        np.testing.assert_array_equal(actual[4], actual[5])


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-11)])
def test_live_scale_vector_prior_preserves_base_draws_and_rng(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    model = _convert(tmp_path, {
        "formula": "y ~ x + z",
        "priors": [
            {"class": "Intercept", "prior": "normal(0, 2)"},
            {"class": "b", "prior": "student_t(4, 0, temp_Intercept)"},
            {"class": "sigma", "prior": "constant(1)"},
        ],
        "data": {"nrows": 3, "columns": [
            {"name": "x", "values": {"kind": "real64", "values": [0, 1, 2]}},
            {"name": "z", "values": {"kind": "real64", "values": [-1, 1, 0]}},
            {"name": "y", "values": {"kind": "real64", "values": [1, 2, 3]}},
        ]},
    })
    sample = _sample(tmp_path, model, """
K_b = kernelof(b, temp_Intercept = temp_Intercept)
s = external(reals)
inputs = s
rstate = rnginit(19)
value, state = rand(K_b(s), rstate)
reference, base_state = rand(iid(StudentT(4), 2), rstate)
after, final_state = rand(Normal(0, 1), state)
base_after, base_final_state = rand(Normal(0, 1), base_state)
outputs = (value, s .* reference, after, base_after, final_state, base_final_state)
""", precision)
    _, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    key = jnp.asarray([19, 0], dtype=jnp.uint64)
    for scale in [.8, 1.7, -1, 0, np.inf, -np.inf, np.nan]:
        actual = sample(key, jnp.asarray(scale, dtype=dtype))
        if math.isfinite(scale) and scale > 0:
            np.testing.assert_allclose(actual[0], actual[1], rtol=tolerance, atol=tolerance)
        else:
            assert np.all(np.isnan(actual[0]))
        np.testing.assert_array_equal(actual[2], actual[3])
        np.testing.assert_array_equal(actual[4], actual[5])

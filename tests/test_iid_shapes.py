"""Multi-axis iid products preserve event shape, density and RNG threading."""

import math

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def _module(tmp_path, source, precision, mode):
    query = tmp_path / "query.flatppl"
    query.write_text(source)
    return ex.emit(query, mode, dtype=precision,
                   restrict_enzyme_compatible=mode != "sample")


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-11)])
@pytest.mark.parametrize("vector_event", [False, True], ids=["scalar", "vector"])
def test_matrix_iid_density_and_parameter_gradients(
    tmp_path, precision, tolerance, vector_event,
):
    _gate_engine("stablehlo")
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    domain = "cartpow(reals, 2)" if vector_event else "reals"
    measure = ("MvNormal([mu, -1.0], eye(2) .* (sigma * sigma))"
               if vector_event else "Normal(mu, sigma)")
    source = _module(tmp_path,
        "mu = external(reals)\nsigma = external(posreals)\n"
        f"x = external(cartpow({domain}, [2, 3]))\n"
        f"z ~ iid({measure}, [2, 3])\nprior = lawof(record(z = z))\n"
        "inputs = (mu, sigma, x)\noutputs = logdensityof(prior, record(z = x))\n",
        precision, "logdensity")
    score = jax.jit(jax.value_and_grad(
        lambda mu, sigma, x: hlo_call(mu, sigma, x, source=source)[0],
        argnums=(0, 1, 2),
    ))
    base = np.array([[.1, -.3, .2], [.4, .2, -.1]], dtype=dtype)
    if vector_event:
        base = np.stack([base, base - 1], axis=-1)
    for mean, scale, offset in [(.25, 1.5, 0), (-.5, .7, .6), (1, 2.3, -1)]:
        mu, sigma = np.asarray(mean, dtype=dtype), np.asarray(scale, dtype=dtype)
        x = np.asarray(base + offset, dtype=dtype)
        residual = x.astype(np.float64) - ([float(mu), -1] if vector_event else float(mu))
        variance = float(sigma)**2
        expected = -x.size * math.log(float(sigma) * math.sqrt(2 * math.pi))
        expected -= .5 * np.square(residual).sum() / variance
        expected_mu = (residual[..., 0] if vector_event else residual).sum() / variance
        expected_sigma = np.square(residual).sum() / float(sigma)**3 - x.size / float(sigma)
        value, gradient = score(jnp.asarray(mu), jnp.asarray(sigma), jnp.asarray(x))
        assert value.devices() == {jax.devices()[0]}
        assert value.dtype == dtype
        np.testing.assert_allclose(value, expected, rtol=tolerance, atol=tolerance)
        for actual, expected_derivative in zip(
            gradient, (expected_mu, expected_sigma, -residual / variance), strict=True,
        ):
            assert actual.devices() == {jax.devices()[0]}
            np.testing.assert_allclose(actual, expected_derivative,
                                       rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("measure,event_shape", [
    ("Normal(0, 1)", ()),
    ("Gamma(2, 3)", ()),
    ("MvNormal([1.0, -1.0], eye(2))", (2,)),
], ids=["normal", "gamma", "mvnormal"])
def test_matrix_iid_sampling_preserves_flat_draws_and_successor(
    tmp_path, precision, measure, event_shape,
):
    _gate_engine("stablehlo")
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def sampler(size):
        source = _module(tmp_path,
            f"x, state = rand(iid({measure}, {size}), rnginit(0))\n"
            "y, advanced = rand(Normal(0, 1), state)\noutputs = (x, y, advanced)\n",
            precision, "sample")
        return jax.jit(lambda key: hlo_call(key, source=source))

    matrix, flat = sampler("[2, 3]"), sampler("6")
    for seed in (0x123456789abcdef, 31):
        key = jnp.asarray([seed, 0], dtype=jnp.uint64)
        shaped, vector = matrix(key), flat(key)
        assert shaped[0].shape == (2, 3) + event_shape
        assert vector[0].shape == (6,) + event_shape
        assert shaped[0].dtype == dtype
        assert np.all(np.isfinite(shaped[0]))
        for actual in shaped:
            assert actual.devices() == {jax.devices()[0]}
        np.testing.assert_array_equal(shaped[0].reshape(vector[0].shape), vector[0])
        np.testing.assert_array_equal(shaped[1], vector[1])
        np.testing.assert_array_equal(shaped[2], vector[2])
        assert not np.array_equal(shaped[2], key)

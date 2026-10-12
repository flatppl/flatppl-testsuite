"""Positive-count laws preserve exact normalizers and conditional gradients."""

import math

import numpy as np
import pytest
from scipy import special, stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def _poisson(rate, count):
    value = stats.poisson.logpmf(count, rate) - stats.poisson.logsf(0, rate)
    derivative = count / rate - 1 - 1 / math.expm1(rate)
    return value, np.array([derivative, 0, 0])


def _negative_binomial(shape, rate, count):
    probability = rate / (1 + rate)
    mass = stats.nbinom.sf(0, shape, probability)
    value = stats.nbinom.logpmf(count, shape, probability) - math.log(mass)
    derivative = [0,
        special.digamma(count + shape) - special.digamma(shape) + math.log(probability) / mass,
        shape / (rate * (1 + rate) * mass) - count / (1 + rate),
    ]
    return value, np.array(derivative)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 4e-5), ("f64", 1e-10)])
def test_positive_count_density_kernel_and_pointwise_gradients(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text("""
p = external(cartpow(reals, 3))
count = external(integers)
inputs = (p, count)
positive_poisson(r) = normalize(truncate(Poisson(r), interval(1, inf)))
positive_nb(a, b) = normalize(truncate(NegativeBinomial(a, b), interval(1, inf)))
r ~ Exponential(1)
a ~ Gamma(2, 1)
b ~ Gamma(2, 1)
y ~ positive_poisson.([r, 2 * r])
z ~ positive_nb.([a, a + 1], b)
K = kernelof(record(y = y, z = z), r = r, a = a, b = b)
point = record(y = [count, count + 1], z = [count, count + 1])
poisson_rows = ((rate, value) -> logdensityof(positive_poisson(rate), value)).([p[1], 2 * p[1]], [count, count + 1])
nb_rows = ((shape, value) -> logdensityof(positive_nb(shape, p[3]), value)).([p[2], p[2] + 1], [count, count + 1])
outputs = (logdensityof(positive_poisson(p[1]), count),
  logdensityof(positive_nb(p[2], p[3]), count),
  logdensityof(K(p[1], p[2], p[3]), point), poisson_rows, nb_rows)
""")
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    count_dtype = np.int32 if precision == "f32" else np.int64

    def scores(point, count):
        poisson, nb, conditional, poisson_rows, nb_rows = hlo_call(point, count, source=source)
        return jnp.concatenate((jnp.stack((poisson, nb, conditional)), poisson_rows, nb_rows))

    values, gradients = jax.jit(scores), jax.jit(jax.jacrev(scores))
    for point, count in [([.01, 2, 50], 1), ([.2, .7, 2], 2), ([3, 2.3, .6], 7), ([20, 4, .05], 3)]:
        point, count = jnp.asarray(point, dtype=dtype), jnp.asarray(count, dtype=count_dtype)
        rate, shape, beta = np.asarray(point).astype(np.float64)
        p, dp = _poisson(rate, int(count))
        p2, dp2 = _poisson(2 * rate, int(count) + 1)
        dp2 *= 2  # The other two components are zero.
        nb, dnb = _negative_binomial(shape, beta, int(count))
        nb2, dnb2 = _negative_binomial(shape + 1, beta, int(count) + 1)
        expected = [p, nb, p + p2 + nb + nb2, p, p2, nb, nb2]
        expected_gradient = [dp, dnb, dp + dp2 + dnb + dnb2, dp, dp2, dnb, dnb2]
        actual, derivative = values(point, count), gradients(point, count)
        assert actual.devices() == derivative.devices() == {jax.devices()[0]}
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(derivative, expected_gradient, rtol=tolerance, atol=tolerance)
    # Positive mass but an excluded count is zero density, not an invalid law.
    point = jnp.asarray([.7, 2, 3], dtype=dtype)
    outside = values(point, jnp.asarray(0, dtype=count_dtype))
    assert np.all(np.isneginf(np.asarray(outside)[[0, 1, 2, 3, 5]]))
    assert np.all(np.isfinite(np.asarray(outside)[[4, 6]]))
    # Poisson(0) has zero retained mass; other invalid parameters are undefined.
    for parameter in range(3):
        for invalid in [0, -2, np.inf, -np.inf, np.nan]:
            affected = [0, 2, 3, 4] if parameter == 0 else [1, 2, 5, 6]
            unaffected = [1, 5, 6] if parameter == 0 else [0, 3, 4]
            if parameter == 1 and invalid == 0:  # The second row has shape + 1.
                affected, unaffected = [1, 2, 5], [0, 3, 4, 6]
            result = values(point.at[parameter].set(invalid), jnp.asarray(1, dtype=count_dtype))
            assert np.all(np.isnan(np.asarray(result)[affected]))
            assert np.all(np.isfinite(np.asarray(result)[unaffected]))

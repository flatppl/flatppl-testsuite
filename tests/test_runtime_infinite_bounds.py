"""Interval masses whose bounds are infinite only at run time keep finite gradients."""

import numpy as np
import pytest
from scipy import stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def _cdf_partials(edge, mu, sigma):
    """d Phi((edge - mu) / sigma) / d(sigma, mu); both vanish at an infinite edge."""
    if np.isinf(edge):
        return 0.0, 0.0
    z = (edge - mu) / sigma
    density = stats.norm.pdf(z)
    return -density * z / sigma, -density / sigma


def _evaluate(source, point, dtype):
    jax, jnp, hlo_call = ex._jax()

    def scores(point):
        return jnp.stack(hlo_call(*point, source=source))

    values, gradients = jax.jit(lambda p: (scores(p), jax.jacrev(scores)(p)))(
        jnp.asarray(point, dtype=dtype))
    return np.asarray(values)[0], np.asarray(gradients)[0]


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-11)])
def test_extlinspace_bin_masses_have_finite_scale_gradients(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "mu = elementof(reals)\nsg = elementof(posreals)\n"
        "lp = logdensityof(BinnedPoissonProcess(extlinspace(0.0, 2.0, 3), "
        "weighted(5.0, Normal(mu, sg))), [1, 2, 3, 1])\n"
        "inputs = (sg, mu)\noutputs = lp\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    dtype = np.float32 if precision == "f32" else np.float64
    edges = np.array([-np.inf, 0.0, 1.0, 2.0, np.inf])
    counts = np.array([1, 2, 3, 1])
    for sigma, mu in [(1.3, 0.4), (0.9, 0.2)]:
        rates = 5 * np.diff(stats.norm.cdf(edges, mu, sigma))
        expected = stats.poisson.logpmf(counts, rates).sum()
        partials = np.array([_cdf_partials(edge, mu, sigma) for edge in edges])
        rate_gradients = 5 * np.diff(partials, axis=0)
        gradient = ((counts / rates - 1)[:, None] * rate_gradients).sum(axis=0)
        value, derivatives = _evaluate(source, [sigma, mu], dtype)
        assert np.all(np.isfinite(derivatives)), derivatives
        np.testing.assert_allclose(value, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(derivatives, gradient, rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-11)])
def test_truncation_bounds_infinite_at_run_time_have_finite_gradients(
        tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "mu = elementof(reals)\nsg = elementof(posreals)\n"
        "a = elementof(reals)\nb = elementof(reals)\n"
        "lp = logdensityof(normalize(truncate(Normal(mu, sg), interval(a, b))), 0.3)\n"
        "inputs = (sg, mu, a, b)\noutputs = lp\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    dtype = np.float32 if precision == "f32" else np.float64
    x = 0.3
    for a, b in [(-np.inf, np.inf), (-0.2, np.inf), (-np.inf, 1.1), (-0.2, 1.1)]:
        sigma, mu = 1.3, 0.4
        expected = stats.truncnorm.logpdf(x, (a - mu) / sigma, (b - mu) / sigma, mu, sigma)
        mass = stats.norm.cdf(b, mu, sigma) - stats.norm.cdf(a, mu, sigma)
        z = (x - mu) / sigma
        upper, lower = _cdf_partials(b, mu, sigma), _cdf_partials(a, mu, sigma)
        bound_density = [0.0 if np.isinf(e) else stats.norm.pdf(e, mu, sigma) for e in (a, b)]
        gradient = [
            (z * z - 1) / sigma - (upper[0] - lower[0]) / mass,
            z / sigma - (upper[1] - lower[1]) / mass,
            bound_density[0] / mass,
            -bound_density[1] / mass,
        ]
        value, derivatives = _evaluate(source, [sigma, mu, a, b], dtype)
        assert np.all(np.isfinite(derivatives)), derivatives
        np.testing.assert_allclose(value, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(derivatives, gradient, rtol=tolerance, atol=tolerance)

"""normalize(truncate(locscale(D, shift, scale), interval(lo, hi))) divides by
P_D((interval - shift) / scale) (§06 locscale, truncate and normalize)."""

import math

import numpy as np
import pytest
from scipy import special, stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def _truncated_normal(x, mu, sigma, lo, hi):
    """Value and (d/dx, d/dmu, d/dsigma) of Normal(mu, |sigma|) truncated to [lo, hi]."""
    z = (x - mu) / sigma
    u = np.array([(lo - mu) / sigma, (hi - mu) / sigma])
    mass = stats.norm.cdf(u[1]) - stats.norm.cdf(u[0])
    value = stats.norm.logpdf(z) - math.log(abs(sigma)) - math.log(abs(mass))
    phi = stats.norm.pdf(u)
    tails = phi * np.where(np.isinf(u), 0.0, u)
    d_mass_mu = (phi[0] - phi[1]) / sigma
    d_mass_sigma = (tails[0] - tails[1]) / sigma
    return value, (
        -z / sigma,
        z / sigma - d_mass_mu / mass,
        z * z / sigma - 1 / sigma - d_mass_sigma / mass,
    )


def _finite_window(x, m, s):
    return _truncated_normal(x, m, s, -0.5, 2.0)


def _half_line_negative_scale(x, m, s):
    return _truncated_normal(x, m, s, 0.0, math.inf)


def _nested(x, m, s):
    # locscale(locscale(Normal(0, 1), 1, 2), m, s) is Normal(m + s, 2 s).
    value, (gx, gmu, gsigma) = _truncated_normal(x, m + s, 2 * s, -1.0, 3.0)
    return value, (gx, gmu, gmu + 2 * gsigma)


CASES = {
    "finite-window": (
        "normalize(truncate(locscale(Normal(0, 1), m, s), interval(-0.5, 2.0)))",
        _finite_window, [(0.3, 0.2, 1.4), (1.9, -1.0, 0.6), (-0.4, 2.5, 3.0)],
    ),
    "half-line-negative-scale": (
        "normalize(truncate(locscale(Normal(0, 1), m, s), interval(0, inf)))",
        _half_line_negative_scale, [(0.3, 0.2, -1.4), (2.0, -1.0, -0.6), (0.1, 2.5, -3.0)],
    ),
    "nested": (
        "normalize(truncate(locscale(locscale(Normal(0, 1), 1, 2), m, s), interval(-1.0, 3.0)))",
        _nested, [(0.3, 0.2, 1.4), (2.9, -1.0, 0.6), (-0.9, 0.5, 2.0)],
    ),
}


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-11)])
def test_truncated_locscale_scores_with_the_preimage_mass(tmp_path, case, precision, tolerance):
    _gate_engine("stablehlo")
    law, oracle, points = CASES[case]
    query = tmp_path / "query.flatppl"
    query.write_text(
        "x = elementof(reals)\nm = elementof(reals)\ns = elementof(reals)\n"
        f"inputs = (x, m, s)\noutputs = logdensityof({law}, x)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    for point in points:
        value, gradient = oracle(*point)
        np.testing.assert_allclose(ex.value(source, list(point)), value,
                                   rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(ex.gradient(source, list(point), [0, 1, 2]), gradient,
                                   rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-6), ("f64", 1e-12)])
def test_half_student_t_prior_matches_scipy(tmp_path, precision, tolerance):
    """The posteriorDB half-t prior: locscale(StudentT(3), 0, 10) on the half line."""
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "x = elementof(reals)\ninputs = x\noutputs = logdensityof("
        "normalize(truncate(locscale(StudentT(3), 0, 10), interval(0, inf))), x)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    for x in [0.2, 1.5, 40.0]:
        expected = math.log(2) + stats.t.logpdf(x, 3, 0, 10)
        slope = -(3 + 1) * x / (3 * 100 + x * x)
        np.testing.assert_allclose(ex.value(source, [x]), expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(ex.gradient(source, [x], [0])[0], slope,
                                   rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-11)])
def test_half_student_t_with_runtime_nu_and_scale(tmp_path, precision, tolerance):
    """F(0) = 1/2 for every nu, and a scale of either sign maps the half line onto a half line."""
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "x = elementof(reals)\nnu = elementof(posreals)\ns = elementof(reals)\n"
        "inputs = (x, nu, s)\noutputs = logdensityof("
        "normalize(truncate(locscale(StudentT(nu), 0, s), interval(0, inf))), x)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    for x, nu, s in [(1.5, 3.0, 10.0), (0.4, 2.5, -1.3), (7.0, 30.0, 2.0)]:
        z = x / s
        expected = math.log(2) + stats.t.logpdf(z, nu) - math.log(abs(s))
        ratio = z * z / nu
        gradient = [
            -(nu + 1) * z / (s * (nu + z * z)),
            0.5 * special.digamma((nu + 1) / 2) - 0.5 * special.digamma(nu / 2) - 0.5 / nu
            - 0.5 * math.log1p(ratio) + 0.5 * (nu + 1) * ratio / (nu * (1 + ratio)),
            ((nu + 1) * z * z / (nu + z * z) - 1) / s,
        ]
        np.testing.assert_allclose(ex.value(source, [x, nu, s]), expected,
                                   rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(ex.gradient(source, [x, nu, s], [0, 1, 2]), gradient,
                                   rtol=tolerance, atol=tolerance)

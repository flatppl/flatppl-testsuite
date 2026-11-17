"""Execute normalized HEP densities against independent numerical references."""

import numpy as np
import pytest
from scipy import integrate, special, stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def _oracle(name, point):
    x, location, scale, *tail = point
    if name == "Landau":
        return stats.landau.logpdf(x, loc=location + scale * np.log(np.pi / 2), scale=scale * np.pi / 2)
    if name == "Voigtian":
        return np.log(special.voigt_profile(x - location, tail[0], scale / 2))
    resonance, slope, power = location, scale, tail[0]
    if not 0 < x < resonance:
        return -np.inf
    u = 1 - (x / resonance)**2
    mass = integrate.quad(lambda t: t**power * np.exp(slope * (t - 1)), 0, 1,
                          epsabs=2e-14, epsrel=2e-13)[0]
    return (np.log(2 * x / resonance**2) + power * np.log(u)
            + slope * (u - 1) - np.log(mass))


def _source(tmp_path, name, precision):
    params = {"Landau": ["loc", "scale"], "Voigtian": ["mean", "width", "sigma"],
              "Argus": ["resonance", "slope", "power"]}[name]
    query = tmp_path / (name + ".flatppl")
    query.write_text(
        'h = standard_module("particle-physics", "0.1")\nx = external(reals)\n'
        + "\n".join(f"{param} = external(reals)" for param in params)
        + f"\nM = h.{name}({', '.join(params)})\n"
        + f"inputs = (x, {', '.join(params)})\noutputs = logdensityof(M, x)\n"
    )
    return ex.emit(query, "logdensity", dtype=precision)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tol", [("f32", 3e-5), ("f64", 2e-12)])
@pytest.mark.parametrize("name", ["Landau", "Voigtian", "Argus"])
def test_hep_densities_runtime_values(tmp_path, name, precision, tol):
    _gate_engine("stablehlo")
    source = _source(tmp_path, name, precision)
    dtype = np.float32 if precision == "f32" else np.float64
    if name == "Landau":
        points = [[x, 0., 1.] for x in [-5., -3., -1., 0., 1., 10., 1000.]]
        points.append([3.2, .4, 1.3])
    elif name == "Voigtian":
        points = [[x, .3, width, sigma] for width, sigma in [(1.2, .7), (.01, 2.), (20., .001)]
                  for x in [-10., .3, 1., 30.]]
    else:
        points = [[x, 2., slope, power] for slope, power in [(-20., .5), (-2., 1.5), (0., .7), (3., 2.), (20., .5)]
                  for x in [.2, 1., 1.9]]
    for point in points:
        point = np.asarray(point, dtype=dtype).astype(float).tolist()
        actual = ex.value(source, point)
        assert actual == pytest.approx(_oracle(name, point), rel=0, abs=tol)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("name,point", [
    ("Landau", [3.2, .4, 1.3]),
    ("Voigtian", [1.3, -.2, 1.2, .7]),
    ("Argus", [1., 2., -2., 1.5]),
    ("Argus", [1., 2., 0., .7]),
    ("Argus", [1., 2., 3., 2.]),
])
def test_hep_density_all_parameter_derivatives(tmp_path, name, point):
    _gate_engine("stablehlo")
    source = _source(tmp_path, name, "f64")
    actual = ex.gradient(source, point, list(range(len(point))))
    for i, derivative in enumerate(actual):
        estimates = []
        for multiplier in (1., .5):
            step = multiplier * 1e-5 * max(abs(point[i]), 1)
            left, right = point.copy(), point.copy()
            left[i] -= step
            right[i] += step
            estimates.append((_oracle(name, right) - _oracle(name, left)) / (2 * step))
        assert estimates[0] == pytest.approx(estimates[1], rel=3e-6, abs=2e-8)
        assert derivative == pytest.approx(estimates[1], rel=3e-6, abs=2e-8)


@pytest.mark.stablehlo_only
def test_argus_support_returns_zero_parameter_adjoints(tmp_path):
    _gate_engine("stablehlo")
    source = _source(tmp_path, "Argus", "f64")
    for x in [-1., 0., 2., 3.]:
        point = [x, 2., 3., .7]
        assert np.isneginf(ex.value(source, point))
        np.testing.assert_array_equal(ex.gradient(source, point, [0, 1, 2, 3]), [0., 0., 0., 0.])

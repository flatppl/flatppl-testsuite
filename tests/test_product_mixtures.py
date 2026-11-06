"""Mixtures of product measures (§06 superpose, joint, normalize)."""

import numpy as np
import pytest
from scipy import stats
from scipy.special import expit

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

ROWS = np.array([[0.2, -0.4], [1.0, 0.5]])
SEEN = np.array([[1, 0], [0, 0]])


def _emit(tmp_path, body, precision):
    query = tmp_path / "query.flatppl"
    query.write_text(body)
    return ex.emit(query, "logdensity", dtype=precision)


def _check(source, oracle, points, tolerance):
    for x in points:
        value, slope = oracle(x)
        np.testing.assert_allclose(ex.value(source, [x]), value, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(ex.gradient(source, [x], [0])[0], slope,
                                   rtol=tolerance, atol=tolerance)


def _joints(a):
    """0.3 Exp(a) x Exp(2) + 0.7 Exp(3) x Exp(4) at (0.5, 0.2)."""
    first = 0.3 * a * np.exp(-a * 0.5) * stats.expon.pdf(0.2, scale=0.5)
    second = 0.7 * stats.expon.pdf(0.5, scale=1 / 3) * stats.expon.pdf(0.2, scale=0.25)
    return np.log(first + second), first * (1 / a - 0.5) / (first + second)


def _capture(omega):
    """Per row: omega times the Bernoulli product plus (1 - omega) times a Dirac at all-false."""
    value = slope = 0.0
    for row, seen in zip(ROWS, SEEN, strict=True):
        product = np.prod(np.where(seen == 1, expit(row), 1 - expit(row)))
        dirac = float((seen == 0).all())
        mass = omega * product + (1 - omega) * dirac
        value += np.log(mass)
        slope += (product - dirac) / mass
    return value, slope


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-6), ("f64", 1e-13)])
def test_a_mixture_of_continuous_joints(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    source = _emit(tmp_path,
        "a = elementof(posreals)\ninputs = a\noutputs = logdensityof(normalize(superpose("
        "weighted(0.3, joint(Exponential(a), Exponential(2))), "
        "weighted(0.7, joint(Exponential(3), Exponential(4))))), [0.5, 0.2])\n", precision)
    _check(source, _joints, [0.6, 1.2, 4.0], tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-6), ("f64", 1e-13)])
def test_a_capture_mixture_over_matrix_cells(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    source = _emit(tmp_path,
        "omega = elementof(unitinterval)\n"
        "capture(row, w) = normalize(superpose(weighted(w, Bernoulli.(invlogit.(row))), "
        "weighted(1 - w, Dirac(fill(false, [1, 2])))))\n"
        "inputs = omega\noutputs = logdensityof("
        "capture.(splitblocks(rowstack([[0.2, -0.4], [1.0, 0.5]]), [1, 2]), omega), "
        "splitblocks(rowstack([[true, false], [false, false]]), [1, 2]))\n", precision)
    _check(source, _capture, [0.2, 0.6, 0.9], tolerance)

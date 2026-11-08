"""The elementary §09 particle-physics distributions on StableHLO.

DoubleSidedCrystalBall, RelativisticBreitWigner and BifurcatedNormal are scored
against §09's own unnormalized formulas, with each normalizer found by quadrature
of that formula (independent of the engine), and the slope in one parameter
against a central difference of the same oracle. In float32 the value may also
move by the log density's change over the rounding of x."""

import numpy as np
import pytest
from scipy.integrate import quad

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def _cb_side(u, a, n):
    a = abs(a)
    if u > a:
        return (n / a) ** n * np.exp(-a * a / 2) * (n / a - a + u) ** (-n)
    return np.exp(-u * u / 2)


def dscb(x, a, m0=5.28, sl=0.02, sr=0.03, ar=2.0, nl=3.0, nr=4.0):
    return _cb_side(-(x - m0) / sl, a, nl) if x <= m0 else _cb_side((x - m0) / sr, ar, nr)


def rbw(x, g, m=91.19):
    return 1.0 / ((x * x - m * m) ** 2 + m * m * g * g) if x > 0 else 0.0


def bif(x, sr, mu=1.0, sl=0.5):
    s = sl if x < mu else sr
    return np.exp(-(x - mu) ** 2 / (2 * s * s))


CASES = {
    "DoubleSidedCrystalBall": ("hep.DoubleSidedCrystalBall(5.28, 0.02, 0.03, a, 2.0, 3.0, 4.0)", dscb,
                               1.5, [5.1, 5.24, 5.27, 5.28, 5.3, 5.4], [(-np.inf, 5.28), (5.28, np.inf)]),
    "RelativisticBreitWigner": ("hep.RelativisticBreitWigner(91.19, a)", rbw,
                                2.49, [80.0, 90.0, 91.19, 93.0, 100.0], [(0, 91.19), (91.19, np.inf)]),
    "BifurcatedNormal": ("hep.BifurcatedNormal(1.0, 0.5, a)", bif,
                         2.0, [-1.0, 0.7, 1.0, 1.3, 5.0], [(-np.inf, 1.0), (1.0, np.inf)]),
}


def _oracle(unnormalized, pieces, x, a):
    mass = sum(quad(lambda t: unnormalized(t, a), lo, hi, limit=400)[0] for lo, hi in pieces)
    return np.log(unnormalized(x, a) / mass)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,rtol", [("f32", 3e-6), ("f64", 1e-8)])
@pytest.mark.parametrize("name", sorted(CASES))
def test_hep_member_matches_its_spec_formula(tmp_path, name, precision, rtol):
    _gate_engine("stablehlo")
    expr, unnormalized, a, xs, pieces = CASES[name]
    query = tmp_path / "query.flatppl"
    query.write_text(
        'hep = standard_module("particle-physics", "0.1")\nx = elementof(reals)\n'
        f"a = elementof(posreals)\nlp = logdensityof({expr}, x)\ninputs = (x, a)\noutputs = lp\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    for x in xs:
        want = _oracle(unnormalized, pieces, x, a)
        # float32 rounds x itself: allow the log density's change over that rounding.
        atol = rtol
        if precision == "f32":
            dx = 1e-6 * max(abs(x), 1.0)
            steep = abs(_oracle(unnormalized, pieces, x + dx, a) - _oracle(unnormalized, pieces, x - dx, a)) / (2 * dx)
            atol += 4 * steep * float(np.spacing(np.float32(x)))
        np.testing.assert_allclose(ex.value(source, [x, a]), want, rtol=rtol, atol=atol)
        h = 1e-5 * a
        slope = (_oracle(unnormalized, pieces, x, a + h) - _oracle(unnormalized, pieces, x, a - h)) / (2 * h)
        np.testing.assert_allclose(ex.gradient(source, [x, a], [1])[0], slope,
                                   rtol=1e-4 if precision == "f32" else 1e-6, atol=1e-4)

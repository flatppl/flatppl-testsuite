"""Check §09 sampler references against independently transcribed densities."""

import itertools
import math

import numpy as np
import pytest
from scipy import integrate

from flatppl_testsuite.sampler_sweep.checks import build_cdf
from flatppl_testsuite.sampler_sweep.oracle import FAMILIES


def _density(slug, x):
    if slug in {"crystalball", "doublecrystalball"}:
        if slug == "crystalball":
            scale, alpha, power = 1.2, 1.5, 8.0
            tail = x < 0.5 - scale * alpha
        else:
            scale, alpha, power = (0.8, 1.4, 8.0) if x <= 0.5 else (1.3, 1.8, 9.0)
            tail = abs(x - 0.5) > scale * alpha
        t = abs((x - 0.5) / scale)
        return (
            math.exp(-(alpha**2) / 2)
            * (power / alpha / (power / alpha - alpha + t)) ** power
            if tail
            else math.exp(-t * t / 2)
        )
    if slug == "argus":
        y = 1 - (x / 3) ** 2
        return x * math.sqrt(y) * math.exp(-2 * y) if 0 < x < 3 else 0.0
    if slug == "relbreitwigner":
        return 1 / ((x * x - 4) ** 2 + 1) if x > 0 else 0.0
    if slug == "voigtian":
        # Independent convolution of the two written normalized densities,
        # rather than the Faddeeva profile used by the CDF implementation.
        return integrate.quad(
            lambda z: (
                math.exp(-z * z / (2 * 0.8**2))
                / (0.8 * math.sqrt(2 * math.pi))
                * 0.6
                / (math.pi * ((x - 0.5 - z) ** 2 + 0.6**2))
            ),
            -np.inf,
            np.inf,
            epsabs=2e-12,
        )[0]
    if slug == "bifurcatednormal":
        scale = 0.8 if x <= 0.5 else 1.3
        return math.sqrt(2 / math.pi) / 2.1 * math.exp(-0.5 * ((x - 0.5) / scale) ** 2)
    raise ValueError(slug)


def _integral(slug, lo, hi, center=0.0, order=0):
    joins = {
        "crystalball": [-1.3],
        "doublecrystalball": [-0.62, 0.5, 2.84],
        "bifurcatednormal": [0.5],
        "relbreitwigner": [2.0],
    }.get(slug, [])
    cuts = [lo] + [x for x in joins if lo < x < hi] + [hi]
    return sum(
        integrate.quad(
            lambda x: (x - center) ** order * _density(slug, x),
            a,
            b,
            epsabs=2e-11,
            epsrel=2e-11,
        )[0]
        for a, b in itertools.pairwise(cuts)
    )


@pytest.mark.parametrize(
    "slug,lo,hi,points",
    [
        ("crystalball", -np.inf, np.inf, [-3.0, 0.5, 2.0]),
        ("doublecrystalball", -np.inf, np.inf, [-2.0, 0.5, 4.0]),
        ("argus", 0.0, 3.0, [0.5, 1.5, 2.8]),
        ("relbreitwigner", 0.0, np.inf, [1.0, 2.0, 4.0]),
        ("bifurcatednormal", -np.inf, np.inf, [-1.0, 0.5, 2.0]),
    ],
)
def test_hep_cdfs_match_written_density_integrals(slug, lo, hi, points):
    family = next(f for f in FAMILIES if f.slug == slug)
    assert family.scipy is not None
    cdf = build_cdf(("dist", *family.scipy))
    mass = _integral(slug, lo, hi)
    expected = [_integral(slug, lo, x) / mass for x in points]
    np.testing.assert_allclose(cdf(np.array(points)), expected, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize(
    "slug,lo,hi",
    [
        ("crystalball", -np.inf, np.inf),
        ("doublecrystalball", -np.inf, np.inf),
        ("argus", 0.0, 3.0),
        ("relbreitwigner", 0.0, np.inf),
        ("bifurcatednormal", -np.inf, np.inf),
    ],
)
def test_hep_moment_bands_use_written_density_moments(slug, lo, hi):
    family = next(f for f in FAMILIES if f.slug == slug)
    mass = _integral(slug, lo, hi)
    mean = _integral(slug, lo, hi, order=1) / mass
    variance = _integral(slug, lo, hi, center=mean, order=2) / mass
    assert mean == pytest.approx(family.mean, rel=2e-10)
    assert variance == pytest.approx(family.var, rel=2e-10)
    if family.fourth is not None:
        fourth = _integral(slug, lo, hi, center=mean, order=4) / mass
        assert fourth == pytest.approx(family.fourth, rel=2e-9)


def test_voigt_cdf_derivative_matches_independent_convolution():
    family = next(f for f in FAMILIES if f.slug == "voigtian")
    assert family.scipy is not None
    cdf = build_cdf(("dist", *family.scipy))
    points = np.array([-2.0, 0.5, 3.0])
    step = 1e-4
    derivative = (cdf(points + step) - cdf(points - step)) / (2 * step)
    np.testing.assert_allclose(
        derivative, [_density("voigtian", x) for x in points], rtol=1e-8
    )
    # Scalar and repeated/unsorted vector requests use the same cumulative law.
    assert cdf(0.5) == 0.5
    np.testing.assert_allclose(
        cdf(np.array([3.0, -2.0, 3.0])), np.asarray(cdf(points))[[2, 0, 2]], atol=2e-13
    )


def test_landau_stable_recipe_retains_spec_location_scale_convention():
    family = next(f for f in FAMILIES if f.slug == "landau")
    assert family.scipy is not None
    cdf = build_cdf(("dist", *family.scipy))
    # Independently evaluated scipy.stats.landau CDF points with
    # loc=log(pi/2), scale=pi/2: CERN's standard density convention.
    standard = np.array([-1.0, 0.0, 1.0, 5.0])
    expected = [
        0.1136146410540233,
        0.28683288012541786,
        0.4510180928195259,
        0.7730267799428359,
    ]
    np.testing.assert_allclose(
        cdf(0.5 + 1.2 * standard), expected, rtol=2e-12, atol=2e-13
    )

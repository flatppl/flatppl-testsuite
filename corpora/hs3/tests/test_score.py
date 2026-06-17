"""Tests for the score stage (Task 3).

score_js.cjs documents logdensityof(obs, record(mu=0, sigma=1)) = -1.7253885332
for the gaussian model, so the absolute check is a real cross-check of the full
node path.  The 2ΔNLL check confirms zero at the reference point.
"""

import pathlib

from flatppl_testsuite.scoring.flatppl_engine import log_density, twice_delta_nll

CONV = pathlib.Path(__file__).parents[1] / "conversions"


def test_gaussian_absolute_logdensity():
    v = log_density(CONV / "gaussian" / "gaussian.flatppl", "obs", {"mu": 0.0, "sigma": 1.0})
    assert abs(v - (-1.7253885332)) < 1e-7


def test_twice_delta_nll_zero_at_reference():
    vec = twice_delta_nll(
        CONV / "gaussian" / "gaussian.flatppl", "obs", "mu",
        [0.0], {"mu": 0.0, "sigma": 1.0},
    )
    assert abs(vec[0]) < 1e-9

"""Grouped Normal likelihoods preserve small residuals in the score and its gradient."""

import json
import math

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def grouped_normal(tmp_path, observations, labels, dtype):
    _gate_engine("stablehlo")
    n = len(observations)
    (tmp_path / "rows.json").write_text(json.dumps({"y": observations, "g": labels}))
    query = tmp_path / "query.flatppl"
    query.write_text(
        f'rows = load_data("rows.json", cartprod(y=cartpow(reals,{n}), g=cartpow(posintegers,{n})))\n'
        "mu = elementof(cartpow(reals,2))\nsigma = elementof(posreals)\n"
        "inputs = (mu, sigma)\noutputs = logdensityof(Normal.(mu[rows.g],sigma),rows.y)\n"
    )
    return ex.emit(query, "logdensity", dtype=dtype)


@pytest.mark.stablehlo_only
def test_normal_residual_energy_at_minimum_float32_scale(tmp_path):
    tiny = 2.0**-126
    sigma = 2 * tiny
    src = grouped_normal(tmp_path, [-tiny, tiny, -tiny, tiny], [1, 1, 2, 2], "f32")
    args = [[0.0, 0.0], sigma]
    # Each standardized residual is ±1/2. Their squared norm is one.
    expected = -4 * (math.log(sigma) + 0.5 * math.log(math.tau)) - 0.5
    assert ex.value(src, args) == pytest.approx(expected, rel=2e-6)
    dmu, dsigma = ex.gradient(src, args, [0, 1])
    np.testing.assert_array_equal(dmu, [0.0, 0.0])
    assert dsigma == pytest.approx(-3 / sigma, rel=2e-6)


@pytest.mark.stablehlo_only
def test_normal_mean_gradient_near_translated_observations(tmp_path):
    center = 1.0 + 2.0**-25
    ulp = math.ulp(center)
    observations = [center] * 254 + [center + ulp]
    src = grouped_normal(tmp_path, observations, [1] * len(observations), "f64")
    sigma = math.sqrt(ulp)
    dmu, = ex.gradient(src, [[center, 0.0], sigma], [0])
    # One observation differs from the active mean by exactly sigma squared.
    np.testing.assert_allclose(dmu, [ulp / sigma**2, 0.0], rtol=2e-13, atol=0)

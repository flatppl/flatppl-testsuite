"""`fill(x, size)` with an array `x` repeats the array (§07)."""

import numpy as np
import pytest
from scipy import stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

Y = np.array([0.3, 0.2, 1.0, 0.9])


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-6), ("f64", 1e-13)])
def test_fill_repeats_an_array_value(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "mu = elementof(reals)\ninputs = mu\noutputs = logdensityof("
        "Normal.(joinblocks(fill([mu, 0.5], 2)), 1), [0.3, 0.2, 1.0, 0.9])\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    for mu in [-0.7, 0.4, 2.0]:
        means = np.array([mu, 0.5, mu, 0.5])
        value = stats.norm.logpdf(Y, means, 1).sum()
        slope = (Y - means)[[0, 2]].sum()
        np.testing.assert_allclose(ex.value(source, [mu]), value, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(ex.gradient(source, [mu], [0])[0], slope,
                                   rtol=tolerance, atol=tolerance)

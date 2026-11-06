"""`filter` over fixed data keeps a static number of elements (§07)."""

import numpy as np
import pytest
from scipy import stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

MASK = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
Y = np.array([0.5, 2.0, 3.5])


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-6), ("f64", 1e-13)])
def test_filter_over_fixed_data_selects_the_kept_cells(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "mask = [1.0, 0.0, 1.0, 1.0, 0.0]\nkeep(k) = mask[k] > 0.0\n"
        "cells = filter(keep, indicesof(zeros(5)))\nmu = elementof(reals)\n"
        "inputs = mu\noutputs = logdensityof(Normal.(mu .* cells, 1), [0.5, 2.0, 3.5])\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    cells = np.flatnonzero(MASK > 0) + 1.0
    for mu in [-0.4, 0.9, 1.7]:
        value = stats.norm.logpdf(Y, mu * cells, 1).sum()
        slope = (cells * (Y - mu * cells)).sum()
        np.testing.assert_allclose(ex.value(source, [mu]), value, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(ex.gradient(source, [mu], [0])[0], slope,
                                   rtol=tolerance, atol=tolerance)

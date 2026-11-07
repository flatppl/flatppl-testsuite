"""A scalar scales a transposed vector in either order (§07 `mul`)."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

W = np.array([1.5, -0.5, 2.0])


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-6), ("f64", 1e-13)])
def test_a_scaled_row_contracts_with_a_vector(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "s = elementof(reals)\nv = [0.3, -1.2, 0.8]\nw = [1.5, -0.5, 2.0]\n"
        "inputs = s\noutputs = logdensityof(Normal((transpose(v) * s) * w + (s * transpose(v)) * w, 1), 0.4)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    v = np.array([0.3, -1.2, 0.8])
    for s in [-0.7, 0.25, 1.9]:
        mean = 2 * s * (v @ W)
        value = -0.5 * np.log(2 * np.pi) - 0.5 * (0.4 - mean) ** 2
        slope = (0.4 - mean) * 2 * (v @ W)
        np.testing.assert_allclose(ex.value(source, [s]), value, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(ex.gradient(source, [s], [0])[0], slope,
                                   rtol=tolerance, atol=tolerance)

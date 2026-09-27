"""Constant tensor branches fold without freezing runtime predicates."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
def test_constant_masks_keep_dynamic_predicate_and_gradient(tmp_path):
    _gate_engine("stablehlo")
    (tmp_path / "model.flatppl").write_text(
        "weights = ifelse.([1, 2, 3] .> 1, [2.0, 3.0, 5.0], [7.0, 8.0, 9.0])\n"
        "absent = [1, 2, 3] .< 0\n"
        "score(x) = sum(weights .* x) + sum(ifelse.(absent, 100.0 * x, x)) "
        "+ ifelse(x > 0.0, x, -2.0 * x)\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        'm = load_module("model.flatppl")\nscore = m.score\n'
        "points = elementof(cartpow(reals, 2))\n"
        "inputs = points\noutputs = sum(score.(points))\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    points = np.array([-2.0, 3.0])
    assert ex.value(source, [points]) == 25.0
    np.testing.assert_allclose(ex.gradient(source, [points], [0]), [[16.0, 19.0]], rtol=0, atol=1e-12)

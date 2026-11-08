"""`Counting(support = S)` on StableHLO (§06): the counting measure on the
integers restricted to `S`, so its log density is 0 on `S ∩ ℤ` and -inf
elsewhere, for an integer point, a real point and a real vector point."""

import math

import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

CASES = [
    ("integers", "interval(0, 10)", [(0, 0.0), (3, 0.0), (10, 0.0), (11, -math.inf), (-1, -math.inf)]),
    ("reals", "interval(0, 10)",
     [(0.0, 0.0), (3.0, 0.0), (3.5, -math.inf), (10.0, 0.0), (10.5, -math.inf), (-1.0, -math.inf)]),
    ("cartpow(reals, 2)", "cartpow(interval(0, 10), 2)",
     [([1.0, 2.0], 0.0), ([1.0, 2.5], -math.inf), ([11.0, 2.0], -math.inf)]),
]


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("point,support,expected", CASES, ids=["integer", "real", "real-vector"])
def test_counting_density_is_its_support_gate(tmp_path, point, support, expected, precision):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        f"x = elementof({point})\nlp = logdensityof(Counting({support}), x)\ninputs = (x)\noutputs = (lp)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    for x, want in expected:
        assert ex.value(source, [x]) == want, x

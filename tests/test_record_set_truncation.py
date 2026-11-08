"""A record-valued Lebesgue truncated by a record set (§03, §06).

A record lies in `cartprod(s = S)` when its field `s` lies in `S`. The
posteriorDB state_space port builds a positive ordered pair this way, since §03
has no positive-ordered set. With log weight -sum(s), the log density is
-sum(s) on positive ordered pairs and -inf elsewhere, and its gradient in s is
-1 per cell."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

QUERY = """log_prior(x) = -sum(x.s)
M = logweighted(log_prior, truncate(Lebesgue(cartprod(s = ordered(2))),
    cartprod(s = cartpow(posreals, 2))))
p = elementof(cartpow(reals, 2))
inputs = (p)
outputs = logdensityof(M, record(s = p))
"""


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_record_set_truncation_scores_inside_and_refuses_outside(tmp_path, precision):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(QUERY)
    source = ex.emit(query, "logdensity", dtype=precision)
    for point in ([0.5, 1.5], [0.25, 3.0]):
        np.testing.assert_allclose(ex.value(source, [point]), -sum(point), rtol=1e-6)
        np.testing.assert_allclose(ex.gradient(source, [point], [0])[0], [-1.0, -1.0])
    # Not ordered, and not positive: outside the truncation set.
    for point in ([1.5, 0.5], [-0.5, 1.5]):
        assert ex.value(source, [point]) == -np.inf

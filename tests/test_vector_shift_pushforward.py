"""A scaled input plus a vector shift scores without an annotation.

§07 gives `mul` a scalar-vector domain and `add` arrays of one shape, so
`a * x + c` over `x ~ iid(Normal(1, 2), 3)` with `c = [1, 0, -1]` is affine
(§06 case 1) and its law is `multivariate_normal(a + c, (2a)^2 I)`.
"""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

# scipy.stats.multivariate_normal(a + c, (2 a)^2 I).logpdf([3, 1, -2]).
SPELLINGS = {
    "fn-scale-then-shift": ("fn(2.0 * _ + c)", -7.228198682973689),
    "lambda-scale-then-shift": ("x -> 2.0 * x + c", -7.228198682973689),
    "fn-shift-then-scale": ("fn(c + 2.0 * _)", -7.228198682973689),
    "fn-divide-then-shift": ("fn(_ / 2.0 + c)", -5.1318155996140185),
}


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-12)])
def test_vector_shift_pushforward_scores(tmp_path, spelling, precision, tolerance):
    _gate_engine("stablehlo")
    forward, expected = SPELLINGS[spelling]
    query = tmp_path / "query.flatppl"
    query.write_text(
        "c = [1.0, 0.0, -1.0]\ny = elementof(cartpow(reals, 3))\n"
        f"lp = logdensityof(pushfwd({forward}, iid(Normal(1.0, 2.0), 3)), y)\n"
        "inputs = y\noutputs = lp\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    dtype = np.float32 if precision == "f32" else np.float64
    value = ex.value(source, [np.array([3.0, 1.0, -2.0], dtype=dtype)])
    np.testing.assert_allclose(value, expected, rtol=tolerance, atol=tolerance)

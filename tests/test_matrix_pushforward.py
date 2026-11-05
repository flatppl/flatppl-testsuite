"""A pushforward over a matrix-variate iid base scores.

`y = 2x` over `x ~ iid(Normal(1, 2), [2, 2])` is Normal(2, 4) in each of the four
cells, and its log-volume counts every cell: 4 log 2 (§06 "Engine contract for
`pushfwd` density evaluation"). §07 gives `mul` a scalar-matrix domain, so the
unannotated `fn(2.0 * _)` is the same map as the annotated bijection.
"""

import numpy as np
import pytest
from scipy import stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

FORWARDS = {
    "explicit-bijection": "bijection(fn(2.0 * _), fn(_ / 2.0), 4.0 * log(2.0))",
    "scalar-scale": "fn(2.0 * _)",
}


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("forward", FORWARDS)
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-12)])
def test_matrix_variate_pushforward_scores(tmp_path, forward, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "y = elementof(cartpow(reals, [2, 2]))\n"
        f"lp = logdensityof(pushfwd({FORWARDS[forward]}, iid(Normal(1.0, 2.0), [2, 2])), y)\n"
        "inputs = y\noutputs = lp\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def score(y):
        return hlo_call(y, source=source)[0]

    evaluate = jax.jit(lambda y: (score(y), jax.grad(score)(y)))
    for y in [np.array([[3.0, 1.0], [-2.0, 0.5]]), np.array([[0.0, -1.5], [4.0, 2.25]])]:
        value, slope = evaluate(jnp.asarray(y, dtype=dtype))
        np.testing.assert_allclose(value, stats.norm(2, 4).logpdf(y).sum(), rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(slope, -(y - 2) / 16, rtol=tolerance, atol=tolerance)

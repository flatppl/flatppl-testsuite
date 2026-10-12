"""Promoted interval endpoints replace their source values on every call."""

import math

import numpy as np
import pytest
from scipy import stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-11)])
def test_promoted_interval_bounds_change_scores_and_gradients(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "lo = -0.3\nhi = inf\nx = 0.4\n"
        "lower = lo\nupper = hi\ninputs = (lo, hi, x)\noutputs = ("
        + ", ".join(
            f"logdensityof(normalize(truncate({law}, interval(lower, upper))), x)"
            for law in ("Beta(2, 3)", "Gamma(2, 1.3)", "StudentT(4)")
        )
        + ")\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision, numerical_integrals=True)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def scores(point):
        return jnp.stack(hlo_call(*point, source=source))

    evaluate = jax.jit(lambda point: (scores(point), jax.jacrev(scores)(point)))
    laws = (stats.beta(2, 3), stats.gamma(2, scale=1 / 1.3), stats.t(4))
    for point in np.array([[-.3, 1.5, .4], [.2, .8, .5], [0, 2, .7]], dtype=dtype):
        lo, hi, x = point.astype(np.float64)
        expected, gradients = [], []
        slopes = (1 / x - 2 / (1 - x), 1 / x - 1.3, -5 * x / (4 + x*x))
        for law, slope in zip(laws, slopes, strict=True):
            mass = law.cdf(hi) - law.cdf(lo)
            expected.append(law.logpdf(x) - math.log(mass))
            gradients.append([law.pdf(lo) / mass, -law.pdf(hi) / mass, slope])
        values, derivatives = evaluate(point)
        assert values.devices() == derivatives.devices() == {jax.devices()[0]}
        np.testing.assert_allclose(values, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(derivatives, gradients, rtol=tolerance, atol=tolerance)

"""Negative binomial log ratios stay accurate where the two logarithms nearly
cancel (§08): NegativeBinomial2 with mu >> psi, NegativeBinomial with large
beta. The slopes are small differences of terms near one, so a difference of
two logarithms loses most of its float32 digits there."""

import numpy as np
import pytest
from scipy import stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

CASES = {
    # d/dmu log NB2(k | mu, psi) = k/mu - (k + psi)/(mu + psi).
    "NegativeBinomial2": (
        "a = elementof(posreals)\nb = elementof(posreals)\ninputs = (a, b)\n"
        "outputs = logdensityof(NegativeBinomial2(mu = a, psi = b), 1005)\n",
        (1000.0, 7.0), 0,
        lambda a, b: stats.nbinom.logpmf(1005, b, b / (a + b)),
        lambda a, b: 1005 / a - (1005 + b) / (a + b),
    ),
    # d/dbeta log NB(k | alpha, beta) = alpha/(beta (beta + 1)) - k/(beta + 1).
    "NegativeBinomial": (
        "a = elementof(posreals)\nb = elementof(posreals)\ninputs = (a, b)\n"
        "outputs = logdensityof(NegativeBinomial(alpha = a, beta = b), 4)\n",
        (5e5, 1e5), 1,
        lambda a, b: stats.nbinom.logpmf(4, a, b / (b + 1)),
        lambda a, b: a / (b * (b + 1)) - 4 / (b + 1),
    ),
}


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("family", sorted(CASES))
@pytest.mark.parametrize("precision,slope_rtol", [("f32", 1e-4), ("f64", 1e-10)])
def test_negative_binomial_slope_where_logs_cancel(tmp_path, family, precision, slope_rtol):
    _gate_engine("stablehlo")
    text, point, argnum, value, slope = CASES[family]
    query = tmp_path / "query.flatppl"
    query.write_text(text)
    source = ex.emit(query, "logdensity", dtype=precision)
    np.testing.assert_allclose(ex.gradient(source, list(point), [argnum])[0], slope(*point),
                               rtol=slope_rtol, atol=0)
    # The value keeps lgamma's own float32 rounding at these large arguments.
    value_atol = 2e-2 if precision == "f32" else 1e-8
    np.testing.assert_allclose(ex.value(source, list(point)), value(*point), rtol=0, atol=value_atol)

"""A Categorical broadcast over a vector of probability vectors scores each
observation with its own vector (§04 broadcasting, §08 Categorical)."""

import numpy as np
import pytest
from scipy.special import log_softmax, softmax

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def _score(tmp_path, body, precision):
    query = tmp_path / "query.flatppl"
    query.write_text(f"a = elementof(reals)\n{body}inputs = a\noutputs = lp\n")
    return ex.emit(query, "logdensity", dtype=precision)


def _gathered_rows(a):
    """rows = [[s, 1 - s], [1 - s, s]], s = 1 / (1 + exp(-a)); rows [1, 2, 2]; y = [1, 1, 2]."""
    s = 1 / (1 + np.exp(-a))
    value = 2 * np.log(s) + np.log1p(-s)
    return value, 2 * (1 - s) - s


def _computed_vectors(a):
    """softmax([a i, 0, -a]) for i = 1..4; y = [1, 3, 2, 1]."""
    value = slope = 0.0
    for i, y in zip([1, 2, 3, 4], [1, 3, 2, 1], strict=True):
        dz = np.array([i, 0.0, -1.0])
        value += log_softmax([a * i, 0.0, -a])[y - 1]
        slope += dz[y - 1] - softmax([a * i, 0.0, -a]) @ dz
    return value, slope


def _zero_based_rows(a):
    """softmax([a, 0, 1]) and softmax([0, a, 2]); rows [2, 1, 2, 2]; y = [0, 2, 1, 2]."""
    zs = [lambda t: [t, 0.0, 1.0], lambda t: [0.0, t, 2.0]]
    dzs = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0])]
    value = slope = 0.0
    for r, y in zip([2, 1, 2, 2], [0, 2, 1, 2], strict=True):
        z = zs[r - 1](a)
        value += log_softmax(z)[y]
        slope += dzs[r - 1][y] - softmax(z) @ dzs[r - 1]
    return value, slope


CASES = {
    "gathered-rows": (
        "s = 1 / (1 + exp(-a))\nrows = [[s, 1 - s], [1 - s, s]]\n"
        "lp = logdensityof(Categorical.(rows[[1, 2, 2]]), [1, 1, 2])\n",
        _gathered_rows,
    ),
    "computed-vectors": (
        "probs(t, i) = softmax([t * i, 0.0, -t])\n"
        "lp = logdensityof(Categorical.(probs.(a, [1, 2, 3, 4])), [1, 3, 2, 1])\n",
        _computed_vectors,
    ),
    "zero-based-rows": (
        "rows = [softmax([a, 0.0, 1.0]), softmax([0.0, a, 2.0])]\n"
        "lp = logdensityof(Categorical0.(rows[[2, 1, 2, 2]]), [0, 2, 1, 2])\n",
        _zero_based_rows,
    ),
}


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-6), ("f64", 1e-13)])
def test_categorical_broadcast_over_probability_vectors(tmp_path, case, precision, tolerance):
    _gate_engine("stablehlo")
    body, oracle = CASES[case]
    source = _score(tmp_path, body, precision)
    for a in [-1.2, 0.3, 2.5]:
        value, slope = oracle(a)
        np.testing.assert_allclose(ex.value(source, [a]), value, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(ex.gradient(source, [a], [0])[0], slope, rtol=tolerance, atol=tolerance)

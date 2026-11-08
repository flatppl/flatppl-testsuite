"""The final parameter of a sequence operation reaches the callback whole (§04
"Explicit callback parameters").

`scan(f, init, xs, p)` calls `f(acc, xs[i], p)` and `kscan(kernel, init, xs, p)`
calls `kernel(state, xs[i], p)`. A one-dimensional `p` with the length of `xs` is
the case that distinguishes "whole" from "traversed with xs": each oracle below is
computed by hand for `p` whole, and differs from the traversed reading."""

import math

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

SCAN_VECTOR = """step(a, x, w) = a + x * sum(w)
xs = elementof(cartpow(reals, 2))
w = elementof(cartpow(reals, 2))
inputs = (xs, w)
outputs = scan(step, 0.0, xs, w)
"""

SCAN_RECORD = """step(a, x, p) = p.s * a + x
xs = elementof(cartpow(reals, 3))
s = elementof(reals)
inputs = (xs, s)
outputs = scan(step, 0.0, xs, record(s = s))
"""

KSCAN_VECTOR = """k(state, x, w) = Normal(state + x, sum(w))
w = elementof(cartpow(posreals, 2))
traj = elementof(cartpow(reals, 2))
inputs = (w, traj)
outputs = logdensityof(kscan(k, 0.0, [1.0, 2.0], w), traj)
"""


def normal_logpdf(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)


def vector_value(source, arguments):
    _, jnp, _ = ex._jax()
    return np.asarray(ex._jitted_value(source)(*ex._arguments(jnp, source, arguments)))


def emit(tmp_path, text, precision):
    query = tmp_path / "query.flatppl"
    query.write_text(text)
    return ex.emit(query, "logdensity", dtype=precision)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,rtol", [("f32", 1e-6), ("f64", 1e-14)])
def test_scan_passes_a_vector_parameter_whole(tmp_path, precision, rtol):
    _gate_engine("stablehlo")
    source = emit(tmp_path, SCAN_VECTOR, precision)
    # sum(w) = 110 at every step: 0 + 1 * 110 = 110, then 110 + 2 * 110 = 330.
    # Traversed with xs it would be 0 + 1 * 10 = 10, then 10 + 2 * 100 = 210.
    out = vector_value(source, [[1.0, 2.0], [10.0, 100.0]])
    np.testing.assert_allclose(out, [110.0, 330.0], rtol=rtol)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,rtol", [("f32", 1e-6), ("f64", 1e-14)])
def test_scan_passes_a_record_parameter_whole(tmp_path, precision, rtol):
    _gate_engine("stablehlo")
    source = emit(tmp_path, SCAN_RECORD, precision)
    # s = 2: 2 * 0 + 1 = 1, 2 * 1 + 2 = 4, 2 * 4 + 3 = 11.
    out = vector_value(source, [[1.0, 2.0, 3.0], 2.0])
    np.testing.assert_allclose(out, [1.0, 4.0, 11.0], rtol=rtol)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,rtol", [("f32", 1e-6), ("f64", 1e-13)])
def test_kscan_passes_a_vector_parameter_whole(tmp_path, precision, rtol):
    _gate_engine("stablehlo")
    source = emit(tmp_path, KSCAN_VECTOR, precision)
    w, traj = [0.5, 0.25], [0.1, 0.2]
    # sigma = sum(w) = 0.75 at both steps; traversed it would be 0.5, then 0.25.
    want = normal_logpdf(0.1, 0.0 + 1.0, 0.75) + normal_logpdf(0.2, 0.1 + 2.0, 0.75)
    np.testing.assert_allclose(ex.value(source, [w, traj]), want, rtol=rtol)
    # d/dw_j = d/dsigma for both j: sum over steps of (z^2 - 1) / sigma.
    z = [(0.1 - 1.0) / 0.75, (0.2 - 2.1) / 0.75]
    slope = sum((zi * zi - 1.0) / 0.75 for zi in z)
    np.testing.assert_allclose(ex.gradient(source, [w, traj], [0])[0], [slope, slope], rtol=10 * rtol)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "step(a, x, p, q) = a + x\nxs = elementof(cartpow(reals, 2))\ninputs = (xs)\n"
            "outputs = scan(step, 0.0, xs, [1.0, 2.0], [3.0, 4.0])\n",
            "`scan`: takes one optional final parameter: `scan(f, init, xs, p)`, with `p` passed "
            "whole to the callback; pack several parameters into one record `p`, and several "
            "sequences into one table",
        ),
        (
            "k(s, x, p, q) = Normal(s + x, 1.0)\ntraj = elementof(cartpow(reals, 2))\n"
            "inputs = (traj)\noutputs = logdensityof(kscan(k, 0.0, [1.0, 2.0], [1.0, 2.0], 0.5), traj)\n",
            "`kscan`: takes one optional final parameter: `kscan(kernel, init, xs, p)`",
        ),
        (
            "step(a, x, p) = a + x * p.w\nxs = elementof(cartpow(reals, 2))\ninputs = (xs)\n"
            "outputs = scan(step, 0.0, xs, p = record(w = 2.0))\n",
            "`scan`: the final parameter is positional: `scan(f, init, xs, p)`; pass several "
            "values as one record `p`, for example `record(a = 2.0)`, not as keyword arguments",
        ),
    ],
    ids=["second-sequence-scan", "second-sequence-kscan", "keyword-parameter"],
)
def test_extra_or_keyword_parameters_are_refused_with_the_section_form(tmp_path, text, expected):
    _gate_engine("stablehlo")
    with pytest.raises(RuntimeError) as error:
        emit(tmp_path, text, "f64")
    message = " ".join(str(error.value).split())
    assert expected in message and "Explicit callback parameters" in message

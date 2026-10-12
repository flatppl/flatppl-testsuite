"""VonMises normalization and concentration derivatives across numeric scales."""

import math

import numpy as np
import pytest
from scipy import special

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 5e-5), ("f64", 1e-11)])
def test_von_mises_concentration_values_and_gradients(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "kappa = external(posreals)\ninputs = (kappa)\n"
        "outputs = (logdensityof(VonMises(0, kappa), 0))\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    score = jax.jit(jax.value_and_grad(lambda k: hlo_call(k, source=source)[0]))
    # Near-uniform laws, both old/new approximation boundaries,
    # and a concentrated law. Scaled I0 avoids overflow in the independent oracle.
    for kappa in np.array([1e-8, .1, 1, 2.3, 3.75, 7.999, 8, 8.001, 100, 1000], dtype=dtype):
        k = float(kappa)
        expected = -math.log(2 * math.pi) - math.log(special.i0e(k))
        derivative = 1 - special.i1e(k) / special.i0e(k)
        value, gradient = score(jnp.asarray(kappa))
        assert value.devices() == gradient.devices() == {jax.devices()[0]}
        np.testing.assert_allclose(value, expected, rtol=0, atol=tolerance)
        np.testing.assert_allclose(gradient, derivative, rtol=0, atol=tolerance)

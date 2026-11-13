"""Normal CDF values retain their lower tails in both emitted CDF routes."""

import numpy as np
import pytest
from scipy.integrate import quad

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 4e-6), ("f64", 3e-14)])
def test_normal_cdf_tail_values_and_derivatives(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "z = elementof(reals)\ninputs = z\n"
        "outputs = [invprobit(z), "
        "builtin_touniform(Normal, record(mu = 1, sigma = 2), 2 * z + 1)]\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()

    def scores(z):
        return hlo_call(z, source=source)[0]

    # -6 loses all probability in f32's 0.5*(1+erf); -9 does so in f64 too.
    points = np.array([-9.0, -6.0, -2.0, 0.0, 2.0])
    expected = np.array([
        quad(lambda t: np.exp(-t*t/2) / np.sqrt(2*np.pi), -np.inf, z,
             epsabs=1e-30, epsrel=2e-13)[0] for z in points
    ])
    density = np.exp(-points**2 / 2) / np.sqrt(2 * np.pi)
    dtype = jnp.float32 if precision == "f32" else jnp.float64
    x = jnp.asarray(points, dtype=dtype)
    values = np.asarray(jax.jit(jax.vmap(scores))(x))
    derivatives = np.asarray(jax.jit(jax.vmap(jax.jacrev(scores)))(x))
    # Zero absolute tolerance prevents a rounded-to-zero tail from passing.
    np.testing.assert_allclose(values, np.broadcast_to(expected[:, None], values.shape),
                               rtol=tolerance, atol=0)
    np.testing.assert_allclose(derivatives, np.broadcast_to(density[:, None], derivatives.shape),
                               rtol=tolerance, atol=0)

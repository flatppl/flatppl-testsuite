"""A scalar-scaled affine map over an iid vector base scores without an annotation.

§06 case 1 recognises "affine maps composed from add/sub/neg/mul/divide (with
nonzero scaling)". §07 gives `mul` a scalar-vector domain, `divide` an
array-scalar domain and `add`/`sub` arrays of one shape. So `a * x + c` over
`x ~ iid(Normal(1, 2), 3)` has the Jacobian `a·I` and log-volume `3 log|a|`, and
its law is Normal(a + c_i, 2|a|) per element.
"""

import numpy as np
import pytest
from scipy import stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

BASE = "iid(Normal(1.0, 2.0), 3)"
SHIFT = np.array([1.0, 0.0, -1.0])

# Each map is (prelude, forward map, scale a, shift c): y = a * x + c.
MAPS = {
    "scale-left": ("", "fn(2.0 * _)", 2.0, 0.0),
    "scale-right": ("", "fn(_ * 2.0)", 2.0, 0.0),
    "negative-scale": ("", "fn(-3.0 * _)", -3.0, 0.0),
    "standardize": ("m = [1.0, 0.0, -1.0]\n", "fn((_ - m) / 2.0)", 0.5, -SHIFT / 2),
    "reflect": ("c = [1.0, 0.0, -1.0]\n", "fn(c - _)", -1.0, SHIFT),
}

POINTS = [np.array([3.0, 1.0, -2.0]), np.array([-0.5, 2.5, 0.25])]


def _law(a, c):
    return stats.norm(a * 1.0 + c, abs(a) * 2.0)


def _run(tmp_path, text, precision):
    query = tmp_path / "query.flatppl"
    query.write_text(text)
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    return jax, jnp, source, hlo_call


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("pushforward", MAPS)
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-12)])
def test_scalar_affine_map_scores(tmp_path, pushforward, precision, tolerance):
    _gate_engine("stablehlo")
    prelude, forward, a, c = MAPS[pushforward]
    jax, jnp, source, hlo_call = _run(
        tmp_path,
        f"y = elementof(cartpow(reals, 3))\n{prelude}"
        f"lp = logdensityof(pushfwd({forward}, {BASE}), y)\ninputs = y\noutputs = lp\n",
        precision,
    )
    dtype = np.float32 if precision == "f32" else np.float64

    def score(y):
        return hlo_call(y, source=source)[0]

    evaluate = jax.jit(lambda y: (score(y), jax.grad(score)(y)))
    law = _law(a, c)
    for y in POINTS:
        value, slope = evaluate(jnp.asarray(y, dtype=dtype))
        np.testing.assert_allclose(value, law.logpdf(y).sum(), rtol=tolerance, atol=tolerance)
        mean, sd = a * 1.0 + c, abs(a) * 2.0
        np.testing.assert_allclose(slope, -(y - mean) / sd**2, rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-12)])
def test_parameter_scale_in_a_kernel(tmp_path, precision, tolerance):
    # A positive scale is a §06 "nonzero scaling". The scale reaches the map as an
    # additional argument (§04 "Additional arguments").
    _gate_engine("stablehlo")
    jax, jnp, source, hlo_call = _run(
        tmp_path,
        "a = elementof(posreals)\ny = elementof(cartpow(reals, 3))\n"
        f"K = s -> pushfwd((x, t) -> t * x, {BASE}, s)\n"
        "lp = logdensityof(likelihoodof(K, y), record(s = a))\n"
        "inputs = (a, y)\noutputs = lp\n",
        precision,
    )
    dtype = np.float32 if precision == "f32" else np.float64

    def score(a, y):
        return hlo_call(a, y, source=source)[0]

    evaluate = jax.jit(lambda a, y: (score(a, y), jax.grad(score, argnums=(0, 1))(a, y)))
    for a in [0.5, 2.0]:
        for y in POINTS:
            value, (d_a, d_y) = evaluate(jnp.asarray(a, dtype=dtype), jnp.asarray(y, dtype=dtype))
            np.testing.assert_allclose(value, _law(a, 0.0).logpdf(y).sum(), rtol=tolerance, atol=tolerance)
            # log N(y; a, 2a) = -log(2a) - (y - a)^2 / (8 a^2) - log(2 pi) / 2
            expected_d_a = np.sum(-1 / a + (y - a) / (4 * a**2) + (y - a) ** 2 / (4 * a**3))
            np.testing.assert_allclose(d_a, expected_d_a, rtol=tolerance, atol=tolerance)
            np.testing.assert_allclose(d_y, -(y - a) / (4 * a**2), rtol=tolerance, atol=tolerance)

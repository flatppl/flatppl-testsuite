"""A bijection's scalar logvolume (§06: "a function or a scalar value") scores."""

import numpy as np
import pytest
from scipy import stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

# Doubling maps Normal(mu, sigma) to Normal(2 mu, 2 sigma); its logvolume is log(2).
CASES = {
    "literal-named": (
        "double = bijection(fn(2.0 * _), fn(_ / 2.0), 0.6931471805599453)\n"
        "lp = logdensityof(pushfwd(double, Normal(1.0, 2.0)), y)\n",
        stats.norm(2, 4),
    ),
    "literal-inline": (
        "lp = logdensityof(pushfwd(bijection(fn(2.0 * _), fn(_ / 2.0), "
        "0.6931471805599453), Normal(1.0, 2.0)), y)\n",
        stats.norm(2, 4),
    ),
    "expression-named": (
        "lv = log(2.0)\ndouble = bijection(fn(2.0 * _), fn(_ / 2.0), lv)\n"
        "lp = logdensityof(pushfwd(double, Normal(2.0, 4.0)), y)\n",
        stats.norm(4, 8),
    ),
    "expression-inline": (
        "lp = logdensityof(pushfwd(bijection(fn(2.0 * _), fn(_ / 2.0), log(2.0)), "
        "Normal(2.0, 4.0)), y)\n",
        stats.norm(4, 8),
    ),
}


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-6), ("f64", 1e-13)])
def test_scalar_logvolume_scores_the_pushforward(tmp_path, case, precision, tolerance):
    _gate_engine("stablehlo")
    body, law = CASES[case]
    query = tmp_path / "query.flatppl"
    query.write_text(f"y = elementof(reals)\n{body}inputs = y\noutputs = lp\n")
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def score(y):
        return hlo_call(y, source=source)[0]

    evaluate = jax.jit(lambda y: (score(y), jax.grad(score)(y)))
    for y in [-1.5, 0.0, 3.0, 7.25]:
        value, slope = evaluate(jnp.asarray(y, dtype=dtype))
        np.testing.assert_allclose(value, law.logpdf(y), rtol=tolerance, atol=tolerance)
        mean, sd = law.mean(), law.std()
        np.testing.assert_allclose(slope, -(y - mean) / sd**2, rtol=tolerance, atol=tolerance)

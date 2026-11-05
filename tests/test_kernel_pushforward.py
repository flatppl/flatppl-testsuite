"""A pushforward inside a kernel body scores through every kernel spelling.

§04 reifies a kernel as `functionof(M, mu = mu)`, as a placeholder boundary
`functionof(M, mu = _m_)`, as a lambda `mu -> M` or as a named function
`K(mu) = M`. All four denote the same kernel, so `likelihoodof(K, y)` at
`mu = m` has the same density. §06 scores `pushfwd(f, M)` at y as
`log densityof(M, f^{-1}(y)) - logvolume(f^{-1}(y))`.
"""

import numpy as np
import pytest
from scipy import stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

KERNELS = {
    "functionof": "mu = elementof(reals)\nK = functionof({body}, mu = mu)\n",
    "placeholder": "K = functionof({ph}, mu = _m_)\n",
    "lambda": "K = mu -> {body}\n",
    "named-function": "K(mu) = {body}\n",
}

# Each law is a closed form in mu: the doubling map on Normal(mu, 2) is
# Normal(2 mu, 4), and exp on Normal(mu, 1.5) is LogNormal(mu, 1.5). The last
# entry is d/dmu of the log density.
MAPS = {
    "doubling-bijection": (
        "b = bijection(fn(2.0 * _), fn(_ / 2.0), log(2.0))\n",
        "pushfwd(b, Normal(mu, 2.0))",
        lambda mu: stats.norm(2 * mu, 4),
        [-1.0, 3.0, 5.5],
        lambda mu, y: (y - 2 * mu) / 8,
    ),
    "doubling-fn": (
        "",
        "pushfwd(fn(2.0 * _), Normal(mu, 2.0))",
        lambda mu: stats.norm(2 * mu, 4),
        [-1.0, 3.0, 5.5],
        lambda mu, y: (y - 2 * mu) / 8,
    ),
    "exp": (
        "",
        "pushfwd(exp, Normal(mu, 1.5))",
        lambda mu: stats.lognorm(s=1.5, scale=np.exp(mu)),
        [0.3, 2.0, 7.5],
        lambda mu, y: (np.log(y) - mu) / 2.25,
    ),
}


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("pushforward", MAPS)
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-12)])
def test_kernel_pushforward_scores(tmp_path, kernel, pushforward, precision, tolerance):
    _gate_engine("stablehlo")
    prelude, body, law, points, d_mu = MAPS[pushforward]
    reified = KERNELS[kernel].format(body=body, ph=body.replace("mu", "_m_"))
    query = tmp_path / "query.flatppl"
    query.write_text(
        "m = elementof(reals)\ny = elementof(reals)\n"
        f"{prelude}{reified}"
        "lp = logdensityof(likelihoodof(K, y), record(mu = m))\n"
        "inputs = (m, y)\noutputs = lp\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def score(m, y):
        return hlo_call(m, y, source=source)[0]

    evaluate = jax.jit(lambda m, y: (score(m, y), jax.grad(score)(m, y)))
    for mu in [-0.5, 1.0]:
        for y in points:
            value, slope = evaluate(jnp.asarray(mu, dtype=dtype), jnp.asarray(y, dtype=dtype))
            np.testing.assert_allclose(value, law(mu).logpdf(y), rtol=tolerance, atol=tolerance)
            np.testing.assert_allclose(slope, d_mu(mu, y), rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-12)])
def test_kernel_pushforward_over_an_iid_base(tmp_path, kernel, precision, tolerance):
    # f and f_inv map the whole variate, and logvolume is the one log-volume
    # element of the whole map: 3 log 2 for y = 2x on R^3.
    _gate_engine("stablehlo")
    body = "pushfwd(b, iid(Normal(mu, 2.0), 3))"
    reified = KERNELS[kernel].format(body=body, ph=body.replace("mu", "_m_"))
    query = tmp_path / "query.flatppl"
    query.write_text(
        "m = elementof(reals)\ny = elementof(cartpow(reals, 3))\n"
        "b = bijection(fn(2.0 * _), fn(_ / 2.0), 3.0 * log(2.0))\n"
        f"{reified}"
        "lp = logdensityof(likelihoodof(K, y), record(mu = m))\n"
        "inputs = (m, y)\noutputs = lp\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def score(m, y):
        return hlo_call(m, y, source=source)[0]

    evaluate = jax.jit(lambda m, y: (score(m, y), jax.grad(score)(m, y)))
    y = np.array([3.0, 1.0, -2.0])
    for mu in [-0.5, 1.0]:
        value, slope = evaluate(jnp.asarray(mu, dtype=dtype), jnp.asarray(y, dtype=dtype))
        expected = stats.norm(2 * mu, 4).logpdf(y).sum()
        np.testing.assert_allclose(value, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(slope, ((y - 2 * mu) / 8).sum(), rtol=tolerance, atol=tolerance)

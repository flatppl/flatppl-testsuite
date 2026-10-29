"""Mixture family cells preserve normalized densities and parameter derivatives."""

import numpy as np
import pytest
from scipy.special import digamma, gammaln, logsumexp

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("shared_scale", [False, True])
def test_normal_mixture_preserves_family_cells_and_parameter_gradients(
    tmp_path, shared_scale
):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "w = external(cartpow(posreals, 3))\n"
        "mus = external(cartpow(reals, [3, 2]))\n"
        f"scales = external(cartpow(posreals, {1 if shared_scale else 3}))\n"
        "covariance(s) = rowstack([[1.0 + s*s, 0.25], [0.25, 2.0 + s*s]])\n"
        "covs = covariance.(scales)\n"
        "mix = normalize(ksuperpose(MvNormal, w)(mu = mus, cov = covs))\n"
        "x = elementof(cartpow(reals, 2))\n"
        "inputs = (w, mus, scales, x)\noutputs = logdensityof(mix, x)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    w = np.array([0.3, 1.2, 0.5])
    mus = np.array([[-1.0, 0.5], [0.2, 1.1], [1.4, -0.7]])
    scales = np.array([0.8]) if shared_scale else np.array([0.6, 0.9, 1.3])
    x = np.array([0.4, -0.2])
    covs = np.array([[[1 + s * s, 0.25], [0.25, 2 + s * s]] for s in scales])
    covs = np.broadcast_to(covs, (3, 2, 2))
    delta = x - mus
    solve = np.linalg.solve(covs, delta[..., None])[..., 0]
    lp = (
        -np.log(2 * np.pi)
        - np.linalg.slogdet(covs)[1] / 2
        - np.sum(delta * solve, axis=1) / 2
    )
    total = logsumexp(np.log(w) + lp)
    responsibility = np.exp(np.log(w) + lp - total)
    expected = total - np.log(w.sum())
    scale_gradient = (
        responsibility
        * scales
        * (
            np.sum(solve * solve, axis=1)
            - np.trace(np.linalg.inv(covs), axis1=1, axis2=2)
        )
    )
    if shared_scale:
        scale_gradient = np.array([scale_gradient.sum()])
    gradients = [
        responsibility / w - 1 / w.sum(),
        responsibility[:, None] * solve,
        scale_gradient,
        -np.sum(responsibility[:, None] * solve, axis=0),
    ]
    values = [w, mus, scales, x]
    np.testing.assert_allclose(
        ex.value(source, values), expected, rtol=2e-12, atol=2e-12
    )
    for actual, target in zip(ex.gradient(source, values, [0, 1, 2, 3]), gradients):
        np.testing.assert_allclose(actual, target, rtol=2e-11, atol=2e-11)


@pytest.mark.stablehlo_only
def test_dirichlet_mixture_preserves_parameter_gradients_and_weight_scaling(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "w = external(cartpow(nonnegreals, 3))\n"
        "alpha = external(cartpow(cartpow(posreals, 3), 3))\n"
        "x = elementof(stdsimplex(3))\n"
        "mix = normalize(ksuperpose(Dirichlet, w)(alpha))\n"
        "inputs = (w, alpha, x)\noutputs = logdensityof(mix, x)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    w = np.array([0.3, 1.2, 0.0])
    alpha = np.array([[0.6, 1.1, 2.3], [1.8, 0.8, 1.2], [2.1, 1.7, 0.9]])
    x = np.array([0.2, 0.3, 0.5])
    lp = gammaln(alpha.sum(axis=1)) - gammaln(alpha).sum(axis=1)
    lp += ((alpha - 1) * np.log(x)).sum(axis=1)
    terms = np.full(3, -np.inf)
    terms[:2] = np.log(w[:2]) + lp[:2]
    total = logsumexp(terms)
    responsibility = np.exp(terms - total)
    alpha_gradient = responsibility[:, None] * (
        digamma(alpha.sum(axis=1))[:, None] - digamma(alpha) + np.log(x)
    )
    point_gradient = (responsibility[:, None] * (alpha - 1) / x).sum(axis=0)
    expected = total - np.log(w.sum())
    for scale in (1.0, 7.0):
        values = [scale * w, alpha, x]
        np.testing.assert_allclose(
            ex.value(source, values), expected, rtol=2e-12, atol=2e-12
        )
        for actual, target in zip(
            ex.gradient(source, values, [1, 2]), [alpha_gradient, point_gradient]
        ):
            np.testing.assert_allclose(actual, target, rtol=2e-11, atol=2e-11)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_shared_rows_support_mapped_indices_and_repeated_adjoint(tmp_path, precision):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "data = external(cartpow(cartpow(reals, 3), 4))\n"
        "indices = elementof(cartpow(posintegers, [2, 3]))\n"
        "pick(i) = data[i]\n"
        "inputs = (data, indices)\noutputs = pick.(indices)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype, index_dtype = (
        (np.float32, np.int32) if precision == "f32" else (np.float64, np.int64)
    )
    data = np.arange(12, dtype=dtype).reshape(4, 3) / 8
    indices = np.array([[4, 1, 4], [2, 3, 1]], dtype=index_dtype)
    weights = np.arange(18, dtype=dtype).reshape(2, 3, 3) - 5
    evaluate = jax.jit(lambda data, indices: hlo_call(data, indices, source=source)[0])
    np.testing.assert_array_equal(evaluate(data, indices), data[indices - 1])
    gradient = jax.jit(
        jax.grad(lambda data: jnp.sum(evaluate(data, indices) * weights))
    )
    expected = np.zeros_like(data)
    np.add.at(expected, indices - 1, weights)
    np.testing.assert_array_equal(gradient(data), expected)

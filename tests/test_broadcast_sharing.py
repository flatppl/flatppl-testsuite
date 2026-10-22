"""Shared cell arrays retain their scope, batch domain, and derivatives."""

import subprocess

import numpy as np
import pytest
from scipy.special import digamma, gammaln

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("count", [1, 4])
def test_shared_cell_arrays_use_runtime_points(tmp_path, count):
    _gate_engine("stablehlo")
    (tmp_path / "model.flatppl").write_text(
        "x = elementof(reals)\n"
        "rates = exp.([x, x + 1.0, x + 2.0])\n"
        "L = likelihoodof(functionof(Normal(sum(rates) + sum(rates .* rates), 1.0)), 0.2)\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        'm = load_module("model.flatppl")\n'
        "score(t) = logdensityof(m.L, record(x = t))\n"
        f"points = elementof(cartpow(reals, {count}))\n"
        "inputs = points\noutputs = sum(score.(points))\n"
    )
    serialized = tmp_path / "lowered.flatppl"
    subprocess.run(
        [str(ex.flatppl_bin()), "determinize", str(query), "--keep", "inputs",
         "--keep", "outputs", "-o", str(serialized)],
        check=True, capture_output=True, text=True,
    )
    source = ex.emit(serialized, "logdensity", dtype="f64")
    points = np.linspace(-1.0, 0.5, count)
    rates = np.exp(points[:, None] + np.arange(3))
    mean = rates.sum(axis=1) + (rates * rates).sum(axis=1)
    expected = (-0.5 * (np.log(2 * np.pi) + (mean - 0.2)**2)).sum()
    gradient = -(mean - 0.2) * (rates.sum(axis=1) + 2 * (rates * rates).sum(axis=1))
    np.testing.assert_allclose(ex.value(source, [points]), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ex.gradient(source, [points], [0]), [gradient], rtol=1e-12, atol=1e-12)


@pytest.mark.stablehlo_only
def test_short_selections_keep_shared_children_and_weighted_adjoint(tmp_path):
    _gate_engine("stablehlo")
    (tmp_path / "model.flatppl").write_text(
        "score(t) = sum(2.0 * [t[12]] * exp(t[1]) * exp(t[3])\n"
        "  + 3.0 * [t[12]] * exp(t[3]) * exp(t[8])\n"
        "  + 4.0 * [t[12]] * exp(t[7]) * exp(t[11])) + sin(exp(t[7]))\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        'm = load_module("model.flatppl")\nscore = m.score\n'
        "points = elementof(cartpow(cartpow(reals, 12), 3))\n"
        "inputs = points\noutputs = score.(points)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, hlo_call = ex._jax()
    evaluate = jax.jit(lambda xs: hlo_call(xs, source=source)[0])
    points = np.arange(36, dtype=float).reshape(3, 12) / 32 - 0.5
    left, right = [0, 2, 6], [2, 7, 10]
    factors = np.array([2, 3, 4]) * np.exp(points[:, left]) * np.exp(points[:, right])
    terms = points[:, 11, None] * factors
    shared = np.exp(points[:, 6])
    expected = terms.sum(axis=1) + np.sin(shared)
    np.testing.assert_allclose(evaluate(points), expected, rtol=1e-12, atol=1e-12)
    weights = np.array([1.0, -2.0, 3.0])
    gradient = jax.jit(jax.grad(lambda xs: jnp.sum(evaluate(xs) * weights)))
    expected_gradient = np.zeros_like(points)
    for i, (a, b) in enumerate(zip(left, right)):
        expected_gradient[:, a] += terms[:, i]
        expected_gradient[:, b] += terms[:, i]
    expected_gradient[:, 11] = factors.sum(axis=1)
    expected_gradient[:, 6] += np.cos(shared) * shared
    np.testing.assert_allclose(
        gradient(points), expected_gradient * weights[:, None], rtol=1e-12, atol=1e-12,
    )


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("dtype,tolerance", [("f32", 2e-5), ("f64", 1e-11)])
def test_special_function_packets_preserve_lanes_and_shared_adjoint(tmp_path, dtype, tolerance):
    _gate_engine("stablehlo")
    (tmp_path / "model.flatppl").write_text(
        'hep = standard_module("particle-physics", "0.1")\n'
        "logp(x, s) = logdensityof(hep.ContinuedPoisson(s), x)\n"
        "f(p, s) = [logp(p[1], s), logp(p[3], s), logp(p[1], s), "
        "logp(p[2], s), loggamma(s), p[1]]\n"
        "batch(q, s) = f.(q, s)\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        'm = load_module("model.flatppl")\nbatch = m.batch\n'
        "points = elementof(cartpow(cartpow(cartpow(posreals, 3), 3), 2))\n"
        "scales = elementof(cartpow(posreals, 2))\n"
        "inputs = (points, scales)\noutputs = batch.(points, scales)\n"
    )
    source = ex.emit(query, "logdensity", dtype=dtype)
    jax, jnp, hlo_call = ex._jax()
    evaluate = jax.jit(lambda xs, ss: hlo_call(xs, ss, source=source)[0])
    numpy_dtype = np.float32 if dtype == "f32" else np.float64
    weights = (np.arange(1, 37) * np.where(np.arange(36) % 3, 1, -2)).reshape(2, 3, 6)
    weights = weights.astype(numpy_dtype)
    gradient = jax.jit(jax.grad(
        lambda xs, ss: jnp.sum(evaluate(xs, ss) * weights), argnums=(0, 1),
    ))
    for shift in (0.0, 0.25):
        points = (np.arange(1, 19).reshape(2, 3, 3) / 2 + shift).astype(numpy_dtype)
        scales = np.array([0.5 + shift, 3.5 + shift], dtype=numpy_dtype)
        # SciPy is independent of the emitted CHLO and its Enzyme adjoint.
        xs, ss = points.astype(np.float64), scales.astype(np.float64)
        g = xs * np.log(ss[:, None, None]) - ss[:, None, None] - gammaln(xs + 1)
        expected = np.stack([
            g[..., 0], g[..., 2], g[..., 0], g[..., 1],
            np.broadcast_to(gammaln(ss)[:, None], xs.shape[:-1]), xs[..., 0],
        ], axis=-1)
        np.testing.assert_allclose(evaluate(points, scales), expected, rtol=tolerance, atol=tolerance)
        psi = np.log(ss[:, None, None]) - digamma(xs + 1)
        point_gradient = np.stack([
            (weights[..., 0] + weights[..., 2]) * psi[..., 0] + weights[..., 5],
            weights[..., 3] * psi[..., 1], weights[..., 1] * psi[..., 2],
        ], axis=-1)
        rate_terms = xs / ss[:, None, None] - 1
        scale_gradient = digamma(ss) * weights[..., 4].sum(axis=1) + (
            (weights[..., 0] + weights[..., 2]) * rate_terms[..., 0]
            + weights[..., 3] * rate_terms[..., 1] + weights[..., 1] * rate_terms[..., 2]
        ).sum(axis=1)
        actual_points, actual_scales = gradient(points, scales)
        np.testing.assert_allclose(actual_points, point_gradient, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(actual_scales, scale_gradient, rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 2e-5), ("f64", 2e-11)])
def test_matrix_products_keep_callback_axes_and_adjoint(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "points = external(cartpow(reals, 4))\n"
        "basis(t) = [t,t*t,1.]\n"
        "left(t) = rowstack([[t,1.,2.],[3.,-1.,t*t]])\n"
        "right(t) = colstack([[1.,t,2.],[t*t,0.,-1.]])\n"
        "inner(t) = transpose([1.,-2.,3.]) * basis(t)\n"
        "outer(t) = [t,1.] * transpose([2.,t*t,-1.])\n"
        "row_matrix(t) = transpose([1.,t,2.]) * right(t)\n"
        "matrix_vector(t) = left(t) * basis(t)\n"
        "matrix_matrix(t) = left(t) * right(t)\n"
        "transposed(t) = transpose(left(t))\n"
        "inputs = points\n"
        "outputs = (inner.(points),outer.(points),row_matrix.(points),"
        "matrix_vector.(points),matrix_matrix.(points),transposed.(points))\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    point = jnp.asarray([-2, -.5, 0, 1.25], dtype=dtype)
    t = np.asarray(point).astype(np.float64)
    one, zero = np.ones_like(t), np.zeros_like(t)

    def vector(*items):
        return np.stack(items, axis=-1)

    def matrix(*rows):
        return np.stack(rows, axis=-2)

    # Expand every matrix entry into a scalar polynomial for an independent oracle.
    expected = [
        t - 2*t*t + 3,
        matrix(vector(2*t, t**3, -t), vector(2*one, t*t, -one)),
        vector(t*t + 5, t*t - 2),
        vector(2*t*t + 2, 3*t),
        matrix(vector(2*t + 4, t**3 - 2), vector(3 - t + 2*t*t, 2*t*t)),
        matrix(vector(t, 3*one), vector(one, -one), vector(2*one, t*t)),
    ]
    derivatives = [
        1 - 4*t,
        matrix(vector(2*one, 3*t*t, -one), vector(zero, 2*t, zero)),
        vector(2*t, 2*t),
        vector(4*t, 3*one),
        matrix(vector(2*one, 3*t*t), vector(-1 + 4*t, 4*t)),
        matrix(vector(one, zero), vector(zero, zero), vector(zero, 2*t)),
    ]
    weights = [np.asarray(i + 1 + np.arange(v.size).reshape(v.shape) / (v.size + 1), dtype=dtype)
               for i, v in enumerate(expected)]
    evaluate = jax.jit(lambda p: hlo_call(p, source=source))
    for got, wanted in zip(evaluate(point), expected, strict=True):
        assert got.devices() == {jax.devices()[0]}
        np.testing.assert_allclose(got, wanted, rtol=tolerance, atol=tolerance)
    gradient = jax.jit(jax.grad(lambda p: sum(
        jnp.sum(v * w) for v, w in zip(evaluate(p), weights, strict=True)
    )))(point)
    wanted = sum(np.sum(w.astype(np.float64) * d, axis=tuple(range(1, d.ndim)))
                 for w, d in zip(weights, derivatives, strict=True))
    assert gradient.devices() == {jax.devices()[0]}
    np.testing.assert_allclose(gradient, wanted, rtol=tolerance, atol=tolerance)

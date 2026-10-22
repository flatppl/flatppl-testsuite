"""Thin-plate splines preserve cubic values, broadcasting and adjoints."""

import itertools

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 2e-5), ("f64", 2e-11)])
def test_thinplate_values_curvature_and_center_gradients(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        'sp = standard_module("splines", "0.1")\n'
        "x = elementof(reals)\n"
        "points = elementof(cartpow(reals, 5))\n"
        "centers = elementof(cartpow(reals, 3))\n"
        "beta = [1.0, -2.0, 1.0, 0.25, -0.5]\n"
        "basis(t, c) = sp.thinplate_basis(t, c)\n"
        "value(t, c) = transpose(beta) * basis(t, c)\n"
        "penalty = sp.thinplate_penalty(centers)\n"
        "energy = transpose(beta) * penalty * beta\n"
        "inputs = (x, points, centers)\n"
        "outputs = (basis(x, centers), basis.(points, [centers]), penalty, "
        "value(x, centers), value.(points, [centers]), energy)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    x = jnp.asarray(0, dtype=dtype)
    points = jnp.asarray([-2, -1, 0, 1, 2], dtype=dtype)
    centers = jnp.asarray([-1, 0, 1], dtype=dtype)
    c = np.asarray(centers)
    p = np.asarray(points)
    radial = np.abs(p[:, None] - c) ** 3 / 12
    basis = np.column_stack((radial, np.ones(5), p))
    penalty = np.zeros((5, 5))
    penalty[:3, :3] = np.abs(c[:, None] - c) ** 3 / 12
    # Piecewise cubic values and the integrated squared second derivative.
    values = np.array([2.25, 1.25, 5 / 12, .25, .25])
    derivatives = np.array([-1., -1., -.5, 0., 0.])
    evaluate = jax.jit(lambda x, p, c: hlo_call(x, p, c, source=source))
    actual = evaluate(x, points, centers)
    for got, wanted in zip(actual, [basis[2], basis, penalty, values[2], values, 2 / 3], strict=True):
        assert got.devices() == {jax.devices()[0]}
        np.testing.assert_allclose(got, wanted, rtol=tolerance, atol=tolerance)

    def loss(x, points, centers):
        *_, scalar, vector, energy = evaluate(x, points, centers)
        return scalar + jnp.sum(vector) + energy

    gradient = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(x, points, centers)
    center_gradient = [-4.5, 0., 4.5]
    for got, wanted in zip(gradient, [derivatives[2], derivatives, center_gradient], strict=True):
        assert got.devices() == {jax.devices()[0]}
        np.testing.assert_allclose(got, wanted, rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
def test_boolean_singleton_broadcast_truth_tables(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "a = elementof(cartpow(booleans, 3))\n"
        "b = elementof(cartpow(booleans, 2))\n"
        "x = addaxes(a, 0, 1)\ny = addaxes(b, 1, 0)\n"
        "inputs = (a, b)\noutputs = (land.(x, y), lor.(x, y), lxor.(x, y))\n"
    )
    source = ex.emit(query, "logdensity")
    jax, jnp, hlo_call = ex._jax()
    evaluate = jax.jit(lambda a, b: hlo_call(a, b, source=source))
    for a in itertools.product([False, True], repeat=3):
        for b in itertools.product([False, True], repeat=2):
            left, right = np.asarray(a), np.asarray(b)
            actual = evaluate(jnp.asarray(left), jnp.asarray(right))
            for got, op in zip(actual, [np.logical_and, np.logical_or, np.logical_xor], strict=True):
                assert got.devices() == {jax.devices()[0]}
                np.testing.assert_array_equal(got, op(left[:, None], right))

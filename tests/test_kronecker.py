"""Kronecker products preserve matrix layout, numeric promotion, and adjoints."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 2e-6), ("f64", 1e-12)])
def test_kronecker_runtime_cells_and_derivatives(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "kron.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "A = external(cartpow(reals, [2, 3]))\n"
        "B = external(cartpow(reals, [3, 2]))\n"
        "inputs = (A, B)\noutputs = e.kron(A = A, B = B)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    a = np.arange(1, 7, dtype=dtype).reshape(2, 3)
    b = np.array([[.5, -1], [2, 3], [-.5, 4]], dtype=dtype)
    call = ex.call_with_status(source)
    args = (jnp.asarray(a), jnp.asarray(b))
    actual = ex.expect_valid(jax.jit(call)(*args))[0]
    np.testing.assert_allclose(actual, np.kron(a, b), rtol=tolerance)

    def total(a, b):
        return jnp.sum(call(a, b)[0][0])

    ga, gb = jax.jit(jax.grad(total, argnums=(0, 1)))(*args)
    np.testing.assert_allclose(ga, np.full_like(a, b.sum()), rtol=tolerance)
    np.testing.assert_allclose(gb, np.full_like(b, a.sum()), rtol=tolerance)
    batched = ex.expect_valid(jax.jit(jax.vmap(call))(jnp.stack(args[:1] * 2),
                                                    jnp.stack(args[1:] * 2)))[0]
    np.testing.assert_allclose(batched, np.stack([np.kron(a, b)] * 2), rtol=tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 2e-6), ("f64", 1e-12)])
def test_kronecker_outer_broadcast_preserves_whole_matrix_arguments(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "kron-broadcast.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "A = external(cartpow(cartpow(reals, [2, 3]), 2))\n"
        "B = external(cartpow(reals, [2, 2]))\n"
        "product(a, b) = e.kron(a, b)\n"
        "inputs = (A, B)\noutputs = (e.kron.(A, [B]), product.(A, [B]))\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    a = np.arange(1, 13, dtype=dtype).reshape(2, 2, 3)
    b = np.array([[.5, 2], [-1, 3]], dtype=dtype)
    actual = ex.expect_valid(jax.jit(ex.call_with_status(source))(jnp.asarray(a), jnp.asarray(b)))
    expected = np.stack([np.kron(cell, b) for cell in a])
    for value in actual:
        np.testing.assert_allclose(value, expected, rtol=tolerance)


@pytest.mark.stablehlo_only
def test_kronecker_promotes_both_operands_without_truncating_values(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "kron-mixed.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "A = external(cartpow(integers, [1, 2]))\n"
        "B = external(cartpow(reals, [2, 1]))\n"
        "inputs = (A, B)\noutputs = e.kron(A, B)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, _ = ex._jax()
    actual = ex.expect_valid(jax.jit(ex.call_with_status(source))(
        jnp.array([[2, 3]], dtype=jnp.int64), jnp.array([[.5], [1.25]], dtype=jnp.float64),
    ))[0]
    np.testing.assert_array_equal(actual, [[1, 1.5], [2.5, 3.75]])


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 4e-6), ("f64", 2e-12)])
def test_kronecker_complex_coordinates_and_derivatives(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "kron-complex.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "Ar = external(cartpow(reals, [2, 3]))\n"
        "Ai = external(cartpow(reals, [2, 3]))\n"
        "Br = external(cartpow(reals, [3, 2]))\n"
        "Bi = external(cartpow(reals, [3, 2]))\n"
        "K = e.kron(complex.(Ar, Ai), complex.(Br, Bi))\n"
        "inputs = (Ar, Ai, Br, Bi)\noutputs = (real.(K), imag.(K))\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    ar = np.arange(1, 7, dtype=dtype).reshape(2, 3)
    ai = np.array([[.2, -.3, 2], [1, .5, -1]], dtype=dtype)
    br = np.array([[.5, -1], [2, 3], [-.5, 4]], dtype=dtype)
    bi = np.array([[1, 2], [.2, -.5], [3, .7]], dtype=dtype)
    call = ex.call_with_status(source)
    args = tuple(jnp.asarray(x) for x in [ar, ai, br, bi])
    re, im = ex.expect_valid(jax.jit(call)(*args))
    expected = np.kron(ar + 1j * ai, br + 1j * bi)
    np.testing.assert_allclose(re, expected.real, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(im, expected.imag, rtol=tolerance, atol=tolerance)

    def total(*args):
        (re, im), _ = call(*args)
        return jnp.sum(re + 2 * im)

    gradients = jax.jit(jax.grad(total, argnums=(0, 1, 2, 3)))(*args)
    expected_gradients = [br.sum() + 2 * bi.sum(), -bi.sum() + 2 * br.sum(),
                          ar.sum() + 2 * ai.sum(), -ai.sum() + 2 * ar.sum()]
    for grad, expected in zip(gradients, expected_gradients):
        np.testing.assert_allclose(grad, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
def test_kronecker_complex_literals_and_outer_broadcast(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "kron-complex-literal.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "A = rowstack([[complex(1, 2), complex(3, 4)], [complex(5, 6), complex(7, 8)]])\n"
        "B = external(cartpow(reals, [2, 1]))\n"
        "K = e.kron.( [A, A], [B])\n"
        "inputs = B\noutputs = (real.(K[1]), imag.(K[1]), real.(K[2]), imag.(K[2]))\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, _ = ex._jax()
    b = jnp.array([[.5], [2]], dtype=jnp.float64)
    outputs = ex.expect_valid(jax.jit(ex.call_with_status(source))(b))
    a = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    expected = np.kron(a, b)
    for re, im in zip(outputs[::2], outputs[1::2]):
        np.testing.assert_array_equal(re, expected.real)
        np.testing.assert_array_equal(im, expected.imag)

"""Execute the pivot convention, singular factors, batching, and LU adjoints."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tol", [("f32", 2e-6), ("f64", 1e-12)])
def test_lu_runtime_pivoting_and_singular_cells(tmp_path, precision, tol):
    _gate_engine("stablehlo")
    query = tmp_path / "lu.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "A = external(cartpow(reals, [3, 3]))\n"
        "f = e.lu(A = A)\ninputs = A\noutputs = (f.P, f.L, f.U)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    matrices = np.array([
        [[0, 2, 1], [3, -1, 2], [1, 5, -3]],
        [[4, 2, 1], [2, 1, 5], [1, 3, 2]],
        [[1, 2, 3], [2, 4, 6], [0, 0, 0]],
        np.zeros((3, 3)),
    ], dtype=dtype)
    call = ex.call_with_status(source)
    for a in matrices:
        p, lower, upper = ex.expect_valid(jax.jit(call)(jnp.asarray(a)))
        np.testing.assert_allclose(p @ a, lower @ upper, rtol=tol, atol=tol)
        np.testing.assert_array_equal(np.diag(lower), np.ones(3))
        np.testing.assert_array_equal(np.triu(lower, 1), np.zeros((3, 3)))
        np.testing.assert_array_equal(np.tril(upper, -1), np.zeros((3, 3)))
        np.testing.assert_array_equal(p @ p.T, np.eye(3))
    batched = ex.expect_valid(jax.jit(jax.vmap(call))(jnp.asarray(matrices)))
    np.testing.assert_allclose(batched[0] @ matrices, batched[1] @ batched[2], rtol=tol, atol=tol)

    def reconstructed_sum(a):
        (p, lower, upper), _ = call(a)
        return jnp.sum(p.T @ lower @ upper)

    gradient = jax.jit(jax.grad(reconstructed_sum))(jnp.asarray(matrices[0]))
    np.testing.assert_allclose(gradient, np.ones((3, 3)), rtol=tol, atol=tol)


@pytest.mark.stablehlo_only
def test_lu_integer_input_retains_fractional_multipliers(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "lu-integer.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "A = external(cartpow(integers, [2, 2]))\n"
        "f = e.lu(A)\ninputs = A\noutputs = (f.P, f.L, f.U)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, _ = ex._jax()
    a = jnp.array([[2, 3], [1, 4]], dtype=jnp.int64)
    p, lower, upper = ex.expect_valid(jax.jit(ex.call_with_status(source))(a))
    assert float(lower[1, 0]) == .5
    np.testing.assert_array_equal(p @ a, lower @ upper)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tol", [("f32", 4e-6), ("f64", 2e-12)])
def test_lu_complex_pivots_coordinates_and_adjoint(tmp_path, precision, tol):
    _gate_engine("stablehlo")
    query = tmp_path / "lu-complex.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "Ar = external(cartpow(reals, [3, 3]))\n"
        "Ai = external(cartpow(reals, [3, 3]))\n"
        "f = e.lu(complex.(Ar, Ai))\ninputs = (Ar, Ai)\n"
        "outputs = (f.P, real.(f.L), imag.(f.L), real.(f.U), imag.(f.U))\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    ar = np.array([[.2, 2, 1], [3, -1, 2], [1, 5, -3]], dtype=dtype)
    ai = np.array([[.5, -2, .1], [1, .3, -3], [-.1, 1, 2]], dtype=dtype)
    call = ex.call_with_status(source)
    evaluate = jax.jit(call)
    for scale in ([1., 1e10, 1e-10] if precision == "f32" else [1., 1e150, 1e-150]):
        re, im = ar * scale, ai * scale
        p, lr, li, ur, ui = ex.expect_valid(evaluate(jnp.asarray(re), jnp.asarray(im)))
        lower, upper = lr + 1j * li, ur + 1j * ui
        np.testing.assert_allclose(p @ (re + 1j * im), lower @ upper, rtol=tol, atol=tol * scale)
        np.testing.assert_array_equal(np.diag(lower), np.ones(3))
        np.testing.assert_array_equal(p @ p.T, np.eye(3))

    def total(re, im):
        (p, lr, li, ur, ui), _ = call(re, im)
        real = p.T @ (lr @ ur - li @ ui)
        imaginary = p.T @ (li @ ur + lr @ ui)
        return jnp.sum(real + 2 * imaginary)

    gr, gi = jax.jit(jax.grad(total, argnums=(0, 1)))(jnp.asarray(ar), jnp.asarray(ai))
    np.testing.assert_allclose(gr, 1, rtol=tol, atol=tol)
    np.testing.assert_allclose(gi, 2, rtol=tol, atol=tol)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("n", [1, 9])
def test_lu_static_loop_boundaries_and_adjoints(tmp_path, n):
    _gate_engine("stablehlo")
    query = tmp_path / "lu-size.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        f"A = external(cartpow(reals, [{n}, {n}]))\n"
        "f = e.lu(A)\ninputs = A\noutputs = (f.P, f.L, f.U)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, _ = ex._jax()
    a = np.random.default_rng(117).normal(size=(n, n))
    call = ex.call_with_status(source)
    p, lower, upper = ex.expect_valid(jax.jit(call)(jnp.asarray(a)))
    np.testing.assert_allclose(p @ a, lower @ upper, rtol=1e-12, atol=1e-12)

    def total(a):
        (p, lower, upper), _ = call(a)
        return jnp.sum(p.T @ lower @ upper)

    gradient = jax.jit(jax.grad(total))(jnp.asarray(a))
    np.testing.assert_allclose(gradient, np.ones_like(a), rtol=1e-12, atol=1e-12)


@pytest.mark.stablehlo_only
def test_lu_outer_broadcast_preserves_matrix_cells(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "lu-broadcast.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "A = external(cartpow(cartpow(reals, [3, 3]), 2))\n"
        "factor(a) = e.lu(a)\nf = e.lu.(A)\ng = factor.(A)\ninputs = A\n"
        "outputs = (f.P, f.L, f.U, g.P, g.L, g.U)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, _ = ex._jax()
    a = np.array([[[4., 2, 1], [2, 1, 5], [1, 3, 2]],
                  [[0., 2, 1], [3, -1, 2], [1, 5, -3]]])
    outputs = ex.expect_valid(jax.jit(ex.call_with_status(source))(jnp.asarray(a)))
    for start in (0, 3):
        p, lower, upper = outputs[start:start + 3]
        np.testing.assert_allclose(p @ a, lower @ upper, rtol=1e-12, atol=1e-12)

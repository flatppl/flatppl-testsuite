"""Execute SVD, numerical rank, and minimum-norm least squares."""

from importlib import import_module

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("shape", [(3, 2), (2, 3)])
@pytest.mark.parametrize("precision,tol", [("f32", 2e-4), ("f64", 2e-11)])
def test_complex_svd_and_minimum_norm_solutions(tmp_path, shape, precision, tol):
    _gate_engine("stablehlo")
    m, n = shape
    query = tmp_path / "svd.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        f"Ar = external(cartpow(reals, [{m}, {n}]))\n"
        f"Ai = external(cartpow(reals, [{m}, {n}]))\n"
        f"br = external(cartpow(reals, {m}))\nbi = external(cartpow(reals, {m}))\n"
        "A = complex.(Ar, Ai)\nb = complex.(br, bi)\nf = e.svd(A)\nx = e.lstsq(A, b)\n"
        "inputs = (Ar, Ai, br, bi)\n"
        "outputs = (real.(f.U), imag.(f.U), f.S, real.(f.V), imag.(f.V), "
        "real.(x), imag.(x), e.rank(A))\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    rng = np.random.default_rng(7145)
    a = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
        np.complex64 if precision == "f32" else np.complex128,
    )
    b = (rng.normal(size=m) + 1j * rng.normal(size=m)).astype(a.dtype)
    call = ex.call_with_status(source)
    evaluate = jax.jit(call)
    for matrix in (a, np.zeros_like(a), np.ones_like(a) * (1 + 2j)):
        args = tuple(jnp.asarray(v, dtype=dtype) for v in (matrix.real, matrix.imag, b.real, b.imag))
        ur, ui, s, vr, vi, xr, xi, rank = ex.expect_valid(evaluate(*args))
        u, v, x = np.asarray(ur) + 1j * np.asarray(ui), np.asarray(vr) + 1j * np.asarray(vi), xr + 1j * xi
        np.testing.assert_allclose((u * s) @ v.conj().T, matrix, rtol=tol, atol=tol)
        np.testing.assert_allclose(u.conj().T @ u, np.eye(min(shape)), rtol=tol, atol=tol)
        np.testing.assert_allclose(v.conj().T @ v, np.eye(min(shape)), rtol=tol, atol=tol)
        np.testing.assert_allclose(np.sort(s), np.sort(np.linalg.svd(matrix, compute_uv=False)), rtol=tol, atol=tol)
        np.testing.assert_allclose(x, np.linalg.lstsq(matrix, b, rcond=None)[0], rtol=tol, atol=tol)
        assert int(rank) == np.linalg.matrix_rank(matrix)

    def nuclear_norm(ar, ai):
        outputs, _ = call(ar, ai, jnp.asarray(b.real), jnp.asarray(b.imag))
        return jnp.sum(outputs[2])

    gr, gi = jax.jit(jax.grad(nuclear_norm, argnums=(0, 1)))(jnp.asarray(a.real), jnp.asarray(a.imag))
    u, _, vh = np.linalg.svd(a, full_matrices=False)
    expected = u @ vh
    np.testing.assert_allclose(gr, expected.real, rtol=tol, atol=tol)
    np.testing.assert_allclose(gi, expected.imag, rtol=tol, atol=tol)


@pytest.mark.stablehlo_only
def test_integer_svd_and_least_squares_promote_fractional_results(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "svd-integer.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "A = external(cartpow(integers, [3, 2]))\nb = external(cartpow(integers, 3))\n"
        "f = e.svd(A)\ninputs = (A, b)\noutputs = (f.U, f.S, f.V, e.lstsq(A, b))\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, _ = ex._jax()
    a = np.array([[1, 0], [0, 2], [1, 1]], dtype=np.int64)
    b = np.array([1, 1, 1], dtype=np.int64)
    u, s, v, x = ex.expect_valid(jax.jit(ex.call_with_status(source))(jnp.asarray(a), jnp.asarray(b)))
    np.testing.assert_allclose((u * s) @ v.T, a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(u.T @ u, np.eye(2), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(x, np.linalg.lstsq(a, b, rcond=None)[0], rtol=1e-12, atol=1e-12)


@pytest.mark.stablehlo_only
def test_rank_reports_numerical_failure_instead_of_returning_zero(tmp_path):
    _gate_engine("stablehlo")
    checkify = import_module("jax.experimental.checkify")

    query = tmp_path / "rank.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "A = external(cartpow(reals, [2, 2]))\ninputs = A\noutputs = e.rank(A)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    assert ex.value(source, [np.eye(2).tolist()]) == 2
    assert ex.value(source, [np.zeros((2, 2)).tolist()]) == 0
    with pytest.raises(checkify.JaxRuntimeError, match="numerical algorithm did not converge"):
        ex.value(source, [[[np.nan, 0], [0, 1]]])


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0)])
@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("complex_input", [False, True])
def test_empty_svd_rank_and_least_squares_from_data_derived_shapes(
    tmp_path, shape, precision, complex_input,
):
    _gate_engine("stablehlo")
    m, n = shape
    rows, cols = str(m) if m else "zero", str(n) if n else "zero"
    query = tmp_path / "svd-empty.flatppl"
    inputs = "A = complex.(Ar, Ar)\nb = complex.(br, br)\n" if complex_input else "A = Ar\nb = br\n"
    outputs = ("real.(f.U), imag.(f.U), f.S, real.(f.V), imag.(f.V), real.(x), imag.(x), e.rank(A)"
               if complex_input else "f.U, f.S, f.V, x, e.rank(A)")
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "zero = lengthof(filter(fn(_ > 2), [1]))\nt = external(reals)\n"
        f"Ar = fill(t, [{rows}, {cols}])\nbr = fill(t, {rows})\n"
        + inputs + f"f = e.svd(A)\nx = e.lstsq(A, b)\ninputs = t\noutputs = ({outputs})\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, _ = ex._jax()
    dtype = jnp.float32 if precision == "f32" else jnp.float64
    call = ex.call_with_status(source)
    actual = ex.expect_valid(jax.jit(call)(dtype(2)))
    expected_shapes = ([(m, 0), (m, 0), (0,), (n, 0), (n, 0), (n,), (n,), ()]
                       if complex_input else [(m, 0), (0,), (n, 0), (n,), ()])
    for value, expected_shape in zip(actual, expected_shapes):
        assert value.shape == expected_shape
        np.testing.assert_array_equal(value, np.zeros(expected_shape))

    def total(t):
        values, _ = call(t)
        return sum(jnp.sum(value) for value in values[:-1])

    assert jax.jit(jax.grad(total))(dtype(2)) == 0

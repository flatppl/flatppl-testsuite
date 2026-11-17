"""Execute general right eigenpairs, ordering domains, and derivatives."""

from importlib import import_module

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def _source(tmp_path, precision, n=3):
    query = tmp_path / "eigen.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        f"Ar = external(cartpow(reals, [{n}, {n}]))\n"
        f"Ai = external(cartpow(reals, [{n}, {n}]))\n"
        "f = e.eigen(A = complex.(Ar, Ai))\ninputs = (Ar, Ai)\n"
        "outputs = (real.(f.values), imag.(f.values), real.(f.vectors), imag.(f.vectors))\n"
    )
    return ex.emit(query, "logdensity", dtype=precision)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tol", [("f32", 2e-4), ("f64", 2e-10)])
def test_general_eigenpairs_and_native_batching(tmp_path, precision, tol):
    _gate_engine("stablehlo")
    source = _source(tmp_path, precision)
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    rng = np.random.default_rng(9418)
    a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    matrices = [a, a.real, np.eye(3), np.zeros((3, 3)), np.roll(np.eye(3), 1, axis=0)]
    call = ex.call_with_status(source)
    evaluate = jax.jit(call)
    for matrix in matrices:
        values = ex.expect_valid(evaluate(jnp.asarray(matrix.real, dtype=dtype),
                                         jnp.asarray(matrix.imag, dtype=dtype)))
        lam = np.asarray(values[0]) + 1j * np.asarray(values[1])
        v = np.asarray(values[2]) + 1j * np.asarray(values[3])
        expected = np.linalg.eigvals(matrix)
        rows, cols = linear_sum_assignment(abs(lam[:, None] - expected[None, :]))
        np.testing.assert_allclose(lam[rows], expected[cols], rtol=tol, atol=tol)
        np.testing.assert_allclose(matrix @ v, v * lam, rtol=tol, atol=tol)
        np.testing.assert_allclose(np.sum(abs(v)**2, axis=0), 1, rtol=tol, atol=tol)
    batch = np.stack(matrices)
    values = ex.expect_valid(jax.jit(jax.vmap(call))(jnp.asarray(batch.real, dtype=dtype),
                                                   jnp.asarray(batch.imag, dtype=dtype)))
    lam = np.asarray(values[0]) + 1j * np.asarray(values[1])
    v = np.asarray(values[2]) + 1j * np.asarray(values[3])
    np.testing.assert_allclose(batch @ v, v * lam[:, None, :], rtol=tol, atol=tol)


@pytest.mark.stablehlo_only
def test_simple_eigenvalue_and_canonical_vector_derivatives(tmp_path):
    _gate_engine("stablehlo")
    source = _source(tmp_path, "f64")
    jax, jnp, _ = ex._jax()
    call = ex.call_with_status(source)
    rng = np.random.default_rng(7135)
    a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    args = jnp.asarray(a.real), jnp.asarray(a.imag)
    values = ex.expect_valid(jax.jit(call)(*args))
    lam = np.asarray(values[0]) + 1j * np.asarray(values[1])
    v = np.asarray(values[2]) + 1j * np.asarray(values[3])
    w = np.linalg.inv(v)
    pivot = np.argmax(abs(v[:, 0]))
    expected = np.zeros((2, 3, 3), complex)
    for coordinate in range(2):
        for row in range(3):
            for col in range(3):
                direction = np.zeros((3, 3), complex)
                direction[row, col] = 1 if coordinate == 0 else 1j
                h = w @ direction @ v
                base = sum((v[:, j] * h[j, 0] / (lam[0] - lam[j]) for j in (1, 2)),
                           np.zeros(3, complex))
                alpha = -np.vdot(v[:, 0], base).real - 1j * base[pivot].imag / v[pivot, 0].real
                expected[coordinate, row, col] = (base + alpha * v[:, 0])[1]
    for component in (0, 1):
        def loss(ar, ai, component=component):
            return call(ar, ai)[0][2 + component][1, 0]

        gr, gi = jax.jit(jax.grad(loss, argnums=(0, 1)))(*args)
        target = expected.real if component == 0 else expected.imag
        np.testing.assert_allclose(gr, target[0], rtol=2e-6, atol=2e-8)
        np.testing.assert_allclose(gi, target[1], rtol=2e-6, atol=2e-8)

        def eigenvalue(ar, ai, component=component):
            return call(ar, ai)[0][component][0]

        gr, gi = jax.jit(jax.grad(eigenvalue, argnums=(0, 1)))(*args)
        coefficient = np.outer(w[0, :], v[:, 0])
        er = coefficient.real if component == 0 else coefficient.imag
        ei = -coefficient.imag if component == 0 else coefficient.real
        np.testing.assert_allclose(gr, er, rtol=2e-7, atol=1e-8)
        np.testing.assert_allclose(gi, ei, rtol=2e-7, atol=1e-8)


@pytest.mark.stablehlo_only
def test_eigen_extrema_require_a_nonempty_real_spectrum(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "extrema.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "A = external(cartpow(reals, [3, 3]))\ninputs = A\n"
        "outputs = (e.eigmin(A), e.eigmax(A))\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, _ = ex._jax()
    evaluate = jax.jit(ex.call_with_status(source))
    similarity = np.array([[1., 2, 1], [0, 1, 3], [1, 0, 1]])
    a = similarity @ np.diag([-2., .5, 3.]) @ np.linalg.inv(similarity)
    np.testing.assert_allclose(ex.expect_valid(evaluate(jnp.asarray(a))), [-2, 3], atol=2e-11)
    # A global tolerance would erase this pair beside the large real eigenvalue.
    a = np.array([[1e10, 0, 0], [0, 0, -1e-8], [0, 1e-8, 0]])
    _, status = evaluate(jnp.asarray(a))
    assert int(status) == 11
    checkify = import_module("jax.experimental.checkify")
    with pytest.raises(checkify.JaxRuntimeError, match="nonempty real spectrum"):
        ex.value(source, [a.tolist()])


@pytest.mark.stablehlo_only
def test_empty_eigen_records_and_undefined_extrema(tmp_path):
    _gate_engine("stablehlo")
    prefix = (
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "n = lengthof(filter(fn(_ > 2), [1]))\nt = external(reals)\n"
        "A = fill(t, [n, n])\nf = e.eigen(A)\ninputs = t\n"
    )
    query = tmp_path / "empty-eigen.flatppl"
    query.write_text(prefix + "outputs = (real.(f.values), imag.(f.values), "
                     "real.(f.vectors), imag.(f.vectors))\n")
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, _ = ex._jax()
    call = ex.call_with_status(source)
    values = ex.expect_valid(jax.jit(call)(jnp.asarray(2.)))
    assert [value.shape for value in values] == [(0,), (0,), (0, 0), (0, 0)]

    def total(t):
        return sum(jnp.sum(value) for value in call(t)[0])

    assert jax.jit(jax.grad(total))(jnp.asarray(2.)) == 0
    query.write_text(prefix + "outputs = e.eigmax(A)\n")
    source = ex.emit(query, "logdensity", dtype="f64")
    values, status = jax.jit(ex.call_with_status(source))(jnp.asarray(2.))
    assert int(status) == 11
    assert np.isnan(values[0])


@pytest.mark.stablehlo_only
def test_nonfinite_eigen_input_reports_failed_iteration(tmp_path):
    _gate_engine("stablehlo")
    source = _source(tmp_path, "f64")
    jax, jnp, _ = ex._jax()
    a = np.eye(3)
    a[0, 0] = np.nan
    values, status = jax.jit(ex.call_with_status(source))(jnp.asarray(a), jnp.zeros((3, 3)))
    assert int(status) == 10
    assert all(np.isnan(value).all() for value in values)


@pytest.mark.stablehlo_only
def test_eigen_broadcast_record_rows_and_complex_fields(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "records.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "Ar = external(cartpow(cartpow(reals, [2, 2]), 2))\n"
        "Ai = external(cartpow(cartpow(reals, [2, 2]), 2))\n"
        "A = [complex.(Ar[1], Ai[1]), complex.(Ar[2], Ai[2])]\n"
        "f = e.eigen.(A)\ng = e.lu.(A)\nh = e.svd.(A)\n"
        "r = (field = f[2].vectors,)\ninputs = (Ar, Ai)\n"
        "outputs = ([real.(f.values[1]), real.(f.values[2])], "
        "[imag.(f.values[1]), imag.(f.values[2])], "
        "[real.(f.vectors[1]), real.(f.vectors[2])], "
        "[imag.(f.vectors[1]), imag.(f.vectors[2])], "
        "real.(f[2].vectors), imag.(r.field), real.(g[2].L), imag.(g[2].L), "
        "real.(h[2].U), imag.(h[2].U))\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, _ = ex._jax()
    a = np.array([[[1 + 1j, 2], [0, 3 - 1j]], [[2, 1j], [1 - 2j, 4 + 1j]]])
    values = ex.expect_valid(jax.jit(ex.call_with_status(source))(jnp.asarray(a.real), jnp.asarray(a.imag)))
    lam = np.asarray(values[0]) + 1j * np.asarray(values[1])
    v = np.asarray(values[2]) + 1j * np.asarray(values[3])
    np.testing.assert_allclose(a @ v, v * lam[:, None, :], atol=2e-11)
    np.testing.assert_allclose(values[4], v[1].real, atol=2e-11)
    np.testing.assert_allclose(values[5], v[1].imag, atol=2e-11)
    assert np.max(abs(values[7])) > .01
    u = np.asarray(values[8]) + 1j * np.asarray(values[9])
    np.testing.assert_allclose(u.conj().T @ u, np.eye(2), atol=2e-11)

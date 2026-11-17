"""Execute complex matrix exponentials and independent Frechet derivatives."""

import numpy as np
import pytest
from scipy.linalg import expm, expm_frechet

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tol", [("f32", 4e-5), ("f64", 2e-11)])
def test_complex_matrix_exp_values_and_frechet_derivatives(tmp_path, precision, tol):
    _gate_engine("stablehlo")
    query = tmp_path / "matrix-exp.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "Ar = external(cartpow(reals, [2, 2]))\n"
        "Ai = external(cartpow(reals, [2, 2]))\n"
        "E = e.matexp(A = complex.(Ar, Ai))\n"
        "inputs = (Ar, Ai)\noutputs = (real.(E), imag.(E))\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    call = ex.call_with_status(source)
    evaluate = jax.jit(call)

    def total(re, im):
        (re, im), _ = call(re, im)
        return jnp.sum(re + 2 * im)

    gradient = jax.jit(jax.grad(total, argnums=(0, 1)))
    for a in [np.zeros((2, 2)), np.array([[.2 + .5j, 2 - .7j], [-1 + .2j, .3 - .4j]]),
              np.array([[3j, 1 + 2j], [0, 3j]]), np.diag([50j, -70j])]:
        ar, ai = np.asarray(a.real, dtype=dtype), np.asarray(a.imag, dtype=dtype)
        a = ar.astype(float) + 1j * ai.astype(float)
        args = jnp.asarray(ar), jnp.asarray(ai)
        re, im = ex.expect_valid(evaluate(*args))
        expected = expm(a)
        np.testing.assert_allclose(re, expected.real, rtol=tol, atol=tol)
        np.testing.assert_allclose(im, expected.imag, rtol=tol, atol=tol)
        gr, gi = gradient(*args)
        expected_gradients = []
        for direction in (1., 1j):
            derivatives = np.empty((2, 2))
            for i in range(2):
                for j in range(2):
                    perturbation = np.zeros((2, 2), dtype=complex)
                    perturbation[i, j] = direction
                    derivative = np.asarray(expm_frechet(a, perturbation, compute_expm=False))
                    derivatives[i, j] = np.sum(derivative.real + 2 * derivative.imag)
            expected_gradients.append(derivatives)
        for actual, expected in zip((gr, gi), expected_gradients):
            np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)


@pytest.mark.stablehlo_only
def test_complex_matrix_exp_outer_broadcast_preserves_matrix_cells(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "matrix-exp-broadcast.flatppl"
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "Ar = external(cartpow(reals, [2, 2]))\n"
        "Ai = external(cartpow(reals, [2, 2]))\n"
        "E = e.matexp.([complex.(Ar, Ai), complex.(2 .* Ar, Ai)])\n"
        "inputs = (Ar, Ai)\n"
        "outputs = (real.(E[1]), imag.(E[1]), real.(E[2]), imag.(E[2]))\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, _ = ex._jax()
    ar = np.array([[.2, 2], [-1, .3]])
    ai = np.array([[.5, -.7], [.2, -.4]])
    outputs = ex.expect_valid(jax.jit(ex.call_with_status(source))(jnp.asarray(ar), jnp.asarray(ai)))
    for k in range(2):
        expected = expm((k + 1) * ar + 1j * ai)
        np.testing.assert_allclose(outputs[2 * k], expected.real, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(outputs[2 * k + 1], expected.imag, rtol=2e-11, atol=2e-11)

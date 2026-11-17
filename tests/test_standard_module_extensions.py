"""Execute standard-module extensions against independent numerical oracles."""

from importlib import import_module

import numpy as np
import pytest
from scipy import integrate, interpolate, special

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def _emit(tmp_path, text, precision="f64"):
    _gate_engine("stablehlo")
    query = tmp_path / "extensions.flatppl"
    query.write_text(text)
    return ex.emit(query, "logdensity", dtype=precision)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tol", [("f32", 3e-5), ("f64", 2e-10)])
def test_spline_values_integrals_and_roughness(tmp_path, precision, tol):
    source = _emit(tmp_path,
        's = standard_module("splines", "0.1")\n'
        'x = external(reals)\ny = external(cartpow(reals, 4))\n'
        'knots = [0., .3, 1.1, 2.]\nt = [0., 0., 0., .7, 2., 2., 2.]\n'
        'inputs = (x, y)\noutputs = (s.interp_linear(x, knots, y), '
        's.interp_cubic(x, knots, y), s.interp_pchip(x, knots, y), '
        's.bspline_basis(x, t, 2), s.bspline_derivative(x, t, 2), '
        's.bspline_integral(.2, 1.8, t, 2), s.bspline_penalty(t, 2, 1))\n', precision)
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    knots = np.array([0., .3, 1.1, 2.], dtype=dtype)
    t = np.array([0., 0., 0., .7, 2., 2., 2.], dtype=dtype)
    y = np.array([1., -.5, 2., .8], dtype=dtype)
    basis = interpolate.BSpline(t.astype(float), np.eye(4), 2)
    derivative = basis.derivative()
    penalty = sum(integrate.quad_vec(
        lambda x: np.outer(derivative(x), derivative(x)), lo, hi,
    )[0] for lo, hi in zip([0., float(t[3])], [float(t[3]), 2.]))
    call = ex.call_with_status(source)
    evaluate = jax.jit(call)
    for x in np.array([0., .2, .6, 1.6, 2.], dtype=dtype):
        values = ex.expect_valid(evaluate(jnp.asarray(x), jnp.asarray(y)))
        expected = [np.interp(x, knots, y), interpolate.CubicSpline(knots, y, bc_type="natural")(x),
                    interpolate.PchipInterpolator(knots, y)(x), basis(x), derivative(x),
                    basis.integrate(float(dtype(.2)), float(dtype(1.8))), penalty]
        for actual, target in zip(values, expected, strict=True):
            np.testing.assert_allclose(actual, target, rtol=tol, atol=tol)
    points = jnp.array([.2, .6, 1.6], dtype=dtype)
    batch = ex.expect_valid(jax.jit(jax.vmap(call, in_axes=(0, None)))(points, jnp.asarray(y)))
    np.testing.assert_allclose(batch[1], interpolate.CubicSpline(knots, y, bc_type="natural")(points),
                               rtol=tol, atol=tol)
    def loss(x, y):
        return call(x, y)[0][1]

    dx, dy = jax.jit(jax.grad(loss, argnums=(0, 1)))(jnp.asarray(dtype(.6)), jnp.asarray(y))
    np.testing.assert_allclose(dx, interpolate.CubicSpline(knots, y, bc_type="natural")(.6, 1),
                               rtol=4 * tol, atol=4 * tol)
    np.testing.assert_allclose(dy, interpolate.CubicSpline(knots, np.eye(4), bc_type="natural")(.6),
                               rtol=4 * tol, atol=4 * tol)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tol", [("f32", 2e-4), ("f64", 2e-9)])
def test_special_complements_log_tails_and_scaled_bessel(tmp_path, precision, tol):
    source = _emit(tmp_path,
        's = standard_module("special-functions", "0.1")\n'
        'a = external(posreals)\nb = external(posreals)\nx = external(posreals)\n'
        'p = external(unitinterval)\nv = external(posreals)\nz = external(posreals)\n'
        'w = external(reals)\ninputs = (a, b, x, p, v, z, w)\n'
        'outputs = (s.gammaincc(a, x), s.betaincc(a, b, p), '
        's.log_gammainc(a, x), s.log_gammaincc(a, x), '
        's.log_betainc(a, b, p), s.log_betaincc(a, b, p), '
        's.beta(a, b), s.logbeta(a, b), real(s.bessel_i_scaled(v, z)), '
        's.bessel_k_scaled(v, z), s.log_bessel_i(v, z), s.log_bessel_k(v, z), s.log_ndtr(w))\n',
        precision)
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    call = ex.call_with_status(source)
    evaluate = jax.jit(call)
    for point in ([2.5, 3.7, 1.3, .3, .7, 2., -4.], [10., 20., 100., .9, 2., 1000., -40.]):
        a, b, x, p, v, z, w = np.asarray(point, dtype=dtype).astype(float)
        expected = [special.gammaincc(a, x), special.betaincc(a, b, p),
                    np.log(special.gammainc(a, x)), np.log(special.gammaincc(a, x)),
                    np.log(special.betainc(a, b, p)), np.log(special.betaincc(a, b, p)),
                    special.beta(a, b), special.betaln(a, b), special.ive(v, z), special.kve(v, z),
                    np.log(special.ive(v, z)) + z, np.log(special.kve(v, z)) - z, special.log_ndtr(w)]
        actual = ex.expect_valid(evaluate(*(jnp.asarray(t, dtype=dtype) for t in point)))
        np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
    point = [2.5, 3.7, 1.3, .3, .7, 2., -4.]

    def loss(*args):
        return call(*args)[0][7]

    da, db = jax.jit(jax.grad(loss, argnums=(0, 1)))(
        *(jnp.asarray(t, dtype=dtype) for t in point))
    expected = special.digamma(np.array(point[:2])) - special.digamma(sum(point[:2]))
    np.testing.assert_allclose([da, db], expected, rtol=4 * tol, atol=4 * tol)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tol", [("f32", 3e-4), ("f64", 2e-10)])
def test_factor_solves_hermitian_spectrum_and_shared_rank_cutoff(tmp_path, precision, tol):
    source = _emit(tmp_path,
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        'A = external(cartpow(reals, [2, 2]))\nB = external(cartpow(reals, [2, 2]))\n'
        'L = lower_cholesky(A)\nf = e.eigh(A)\ninputs = (A, B)\n'
        'outputs = (e.solve_triangular(L, B), e.cholesky_solve(L, B), '
        'e.lu_solve(e.lu(A), B), f.values, f.vectors, e.eigvalsh(A), '
        'e.pinv(A = A, rtol = .2), e.rank(A = A, rtol = .2), '
        'e.lstsq(A = A, b = B, rtol = .2), e.cond(A))\n', precision)
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    a = np.array([[4., 1.], [1., 1.]], dtype=dtype)
    b = np.array([[1., -2.], [3., 4.]], dtype=dtype)
    call = ex.call_with_status(source)
    values = ex.expect_valid(jax.jit(call)(jnp.asarray(a), jnp.asarray(b)))
    pinv = np.linalg.pinv(a, rcond=.2)
    expected = [np.linalg.solve(np.linalg.cholesky(a), b), np.linalg.solve(a, b),
                np.linalg.solve(a, b), np.linalg.eigvalsh(a)]
    for actual, target in zip(values[:4], expected, strict=True):
        np.testing.assert_allclose(actual, target, rtol=tol, atol=tol)
    vectors = np.asarray(values[4])
    np.testing.assert_allclose(a @ vectors, vectors * values[3], rtol=tol, atol=tol)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(2), rtol=tol, atol=tol)
    for actual, target in zip(values[5:], [expected[3], pinv, 1, pinv @ b, np.linalg.cond(a)], strict=True):
        np.testing.assert_allclose(actual, target, rtol=tol, atol=tol)

    def loss(a, b):
        return jnp.sum(call(a, b)[0][1])

    da, db = jax.jit(jax.grad(loss, argnums=(0, 1)))(jnp.asarray(a), jnp.asarray(b))
    rhs_gradient = np.linalg.solve(a.T, np.ones_like(b))
    np.testing.assert_allclose(db, rhs_gradient, rtol=4 * tol, atol=4 * tol)
    # Core Cholesky differentiates the symmetric projection of its matrix.
    full_gradient = -rhs_gradient @ np.linalg.solve(a, b).T
    expected_gradient = (full_gradient + full_gradient.T) / 2
    np.testing.assert_allclose(da, expected_gradient, rtol=4 * tol, atol=4 * tol)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("module,expression,shape,invalid", [
    ("special-functions", "m.log_ndtr(x)", "reals", np.nan),
    ("splines", 'm.interp_linear(x, [0., 1.], [0., 2.], "error")', "reals", 2.),
    ("ext-linear-algebra", "m.eigvalsh(x)", "cartpow(reals, [2, 2])", [[1., 2.], [0., 1.]]),
])
def test_standard_module_domains_raise_checked_runtime_errors(tmp_path, module, expression, shape, invalid):
    source = _emit(tmp_path, f'm = standard_module("{module}", "0.1")\n'
                   f'x = external({shape})\ninputs = x\noutputs = {expression}\n')
    jax, jnp, _ = ex._jax()
    _, status = jax.jit(ex.call_with_status(source))(jnp.asarray(invalid))
    assert int(status) == 12
    checkify = import_module("jax.experimental.checkify")
    with pytest.raises(checkify.JaxRuntimeError, match="outside their declared domain"):
        ex.value(source, [invalid])


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tiny,tol", [("f32", 1e-8, 2e-5), ("f64", 1e-18, 2e-12)])
def test_zero_cutoff_preserves_tiny_singular_vector_phase(tmp_path, precision, tiny, tol):
    source = _emit(tmp_path,
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        'A = external(cartpow(reals, [2, 2]))\ninputs = A\n'
        'outputs = (e.rank(A, 0.), e.pinv(A, 0.), e.lstsq(A, eye(2), 0.))\n', precision)
    jax, jnp, _ = ex._jax()
    dtype = jnp.float32 if precision == "f32" else jnp.float64
    rank, inverse, solution = ex.expect_valid(jax.jit(ex.call_with_status(source))(
        jnp.asarray(np.diag([1., -tiny]), dtype=dtype)))
    assert int(rank) == 2
    for actual in (inverse, solution):
        np.testing.assert_allclose(actual, np.diag([1., -1 / tiny]), rtol=tol, atol=tol)


@pytest.mark.stablehlo_only
def test_zero_authored_rank_keeps_a_separate_runtime_status_slot(tmp_path):
    source = _emit(tmp_path,
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        'zero = lengthof(filter(fn(_ > 2), [1]))\nt = external(reals)\n'
        'inputs = t\noutputs = e.rank(fill(t, [zero, zero]))\n')
    jax, jnp, _ = ex._jax()
    values = ex.expect_valid(jax.jit(ex.call_with_status(source))(jnp.asarray(1.)))
    assert len(values) == 1
    assert int(values[0]) == 0


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tiny,tol", [("f32", 1e-24, 3e-4), ("f64", 1e-200, 2e-10)])
@pytest.mark.parametrize("complex_input", [False, True])
def test_correlated_tiny_singular_modes_do_not_underflow(tmp_path, precision, tiny, tol, complex_input):
    source = _emit(tmp_path,
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        'Ar = external(cartpow(reals, [3, 3]))\nAi = external(cartpow(reals, [3, 3]))\n'
        'A = complex.(Ar, Ai)\nP = e.pinv(A, 0.)\nX = e.lstsq(A, eye(3), 0.)\n'
        'inputs = (Ar, Ai)\noutputs = (real.(P), imag.(P), real.(X), imag.(X), '
        'e.cond(A), e.rank(A, 0.), e.rank(A))\n', precision)
    jax, jnp, _ = ex._jax()
    dtype = jnp.float32 if precision == "f32" else jnp.float64
    z = 1 + 1j if complex_input else 1
    block = np.array([[1., z], [0., 1.]])
    a = np.zeros((3, 3), dtype=complex)
    a[0, 0] = 1
    a[1:, 1:] = tiny * block
    values = ex.expect_valid(jax.jit(ex.call_with_status(source))(
        jnp.asarray(a.real, dtype=dtype), jnp.asarray(a.imag, dtype=dtype)))
    expected = np.array([[1., -z], [0., 1.]])
    for offset in (0, 2):
        inverse = np.asarray(values[offset]) + 1j * np.asarray(values[offset + 1])
        np.testing.assert_allclose(inverse[0], [1., 0., 0.], rtol=tol, atol=tol)
        np.testing.assert_allclose(tiny * inverse[1:, 1:], expected, rtol=tol, atol=tol)
    np.testing.assert_allclose(float(values[4]) * tiny,
                               1 / np.linalg.svd(block, compute_uv=False)[-1], rtol=tol)
    assert int(values[5]) == 3
    assert int(values[6]) == 1


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_rejected_special_iteration_reports_nonconvergence(tmp_path, precision):
    source = _emit(tmp_path,
        's = standard_module("special-functions", "0.1")\n'
        'a = external(posreals)\ninputs = a\noutputs = s.log_betainc(a, a, .5)\n', precision)
    jax, jnp, _ = ex._jax()
    dtype = jnp.float32 if precision == "f32" else jnp.float64
    values, status = jax.jit(ex.call_with_status(source))(jnp.asarray(1e20, dtype=dtype))
    assert int(status) == 10
    assert np.isnan(values[0])

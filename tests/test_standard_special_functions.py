"""Special functions execute at live inputs and preserve Bessel order derivatives."""

import numpy as np
import pytest
from scipy import special

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("name,oracle,point", [
    ("gammainc", special.gammainc, [2.5, 1.3]),
    ("gammainc", special.gammainc, [10000., 9999.]),
    ("betainc", special.betainc, [2.5, 3.7, .3]),
    ("betainc", special.betainc, [10000., 20000., .3333]),
])
def test_incomplete_ratios_keep_all_parameter_derivatives(tmp_path, name, oracle, point):
    _gate_engine("stablehlo")
    query = tmp_path / (name + "-gradient.flatppl")
    args = [f"x{i}" for i in range(len(point))]
    query.write_text(
        's = standard_module("special-functions", "0.1")\n'
        + "\n".join(f"{arg} = external(posreals)" for arg in args)
        + f"\ninputs = ({', '.join(args)})\noutputs = s.{name}({', '.join(args)})\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    actual = ex.gradient(source, point, list(range(len(point))))
    for i, value in enumerate(actual):
        estimates = []
        for multiplier in (1., .5):
            step = multiplier * 1e-5 * max(abs(point[i]), 1)
            left, right = point.copy(), point.copy()
            left[i] -= step
            right[i] += step
            estimates.append((oracle(*right) - oracle(*left)) / (2 * step))
        assert estimates[0] == pytest.approx(estimates[1], rel=2e-5, abs=1e-10)
        assert value == pytest.approx(estimates[1], rel=2e-5, abs=1e-10)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 4e-5), ("f64", 2e-10)])
def test_special_functions_preserve_principal_branches_and_runtime_arguments(
    tmp_path, precision, tolerance,
):
    _gate_engine("stablehlo")
    query = tmp_path / "special-functions.flatppl"
    query.write_text(
        's = standard_module("special-functions", "0.1")\n'
        "v = external(reals)\nz = external(reals)\nx = external(posreals)\n"
        "a = external(posreals)\nb = external(posreals)\np = external(unitinterval)\n"
        "j = s.bessel_j(v, z)\ni = s.bessel_i(v, z)\n"
        "inputs = (v, z, x, a, b, p)\n"
        "outputs = (real(j), imag(j), real(i), imag(i), "
        "s.bessel_y(v, x), s.bessel_k(v, x), s.airy(z), "
        "s.gammainc(a, x), s.betainc(a, b, p))\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, _ = ex._jax()
    evaluate = jax.jit(ex.call_with_status(source))
    dtype = np.float32 if precision == "f32" else np.float64
    for point in [
        [.5, -2, 2, 2.5, 3.7, .3],
        [-2.3, 3.2, .1, .1, 2, .0001],
        [12, 10, 30, 10, 50, .95],
    ]:
        point = np.asarray(point, dtype=dtype)
        v, z, x, a, b, p = point.astype(float)
        j, i = special.jv(v, complex(z)), special.iv(v, complex(z))
        expected = np.array([
            j.real, j.imag, i.real, i.imag, special.yv(v, x), special.kv(v, x),
            special.airy(z)[0], special.gammainc(a, x), special.betainc(a, b, p),
        ])
        actual = np.asarray(ex.expect_valid(evaluate(*(jnp.asarray(t) for t in point))))
        # Complex components may vanish exactly. Scale their absolute error
        # to the magnitude of the corresponding complex value.
        atol = np.zeros(9)
        atol[:2] = tolerance * abs(j)
        atol[2:4] = tolerance * abs(i)
        assert np.all(np.abs(actual - expected) <= tolerance * np.abs(expected) + atol)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("name,oracle", [
    ("bessel_j", special.jv), ("bessel_y", special.yv),
    ("bessel_i", special.iv), ("bessel_k", special.kv),
])
def test_bessel_order_derivatives_remain_continuous_at_integers(tmp_path, name, oracle):
    _gate_engine("stablehlo")
    query = tmp_path / (name + ".flatppl")
    expression = f"s.{name}(v, x)"
    if name in ("bessel_j", "bessel_i"):
        expression = f"real({expression})"
    query.write_text(
        's = standard_module("special-functions", "0.1")\n'
        "v = external(reals)\nx = external(posreals)\n"
        f"inputs = (v, x)\noutputs = {expression}\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    step = 1e-5
    for order in (-2., 0., 2.):
        point = [order, 1.3]
        expected = (oracle(order + step, point[1]) - oracle(order - step, point[1])) / (2 * step)
        actual = ex.gradient(source, point, [0])[0]
        assert actual == pytest.approx(expected, rel=2e-6, abs=2e-8)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("name,oracle", [("bessel_j", special.jv), ("bessel_i", special.iv)])
def test_negative_argument_bessel_keeps_phase_derivatives(tmp_path, name, oracle):
    _gate_engine("stablehlo")
    query = tmp_path / (name + "-phase.flatppl")
    query.write_text(
        's = standard_module("special-functions", "0.1")\n'
        "v = external(reals)\ninputs = v\n"
        f"outputs = imag(s.{name}(v, -1.3))\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    step = 1e-5
    for order in (-2., 0., 2.):
        expected = (oracle(order + step, -1.3 + 0j).imag
                    - oracle(order - step, -1.3 + 0j).imag) / (2 * step)
        assert ex.gradient(source, [order], [0])[0] == pytest.approx(expected, rel=2e-6, abs=2e-8)


@pytest.mark.stablehlo_only
def test_airy_preserves_the_derivative_at_zero(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "airy-origin.flatppl"
    query.write_text(
        's = standard_module("special-functions", "0.1")\n'
        "x = external(reals)\ninputs = x\noutputs = s.airy(x)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    for point in (-1e-8, 0., 1e-8):
        expected, derivative, _, _ = special.airy(point)
        assert ex.value(source, [point]) == pytest.approx(expected, rel=2e-11)
        assert ex.gradient(source, [point], [0])[0] == pytest.approx(derivative, rel=2e-10)

"""The pyhf standard helpers agree with their portable FlatPPL definitions."""

from pathlib import Path
import subprocess

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from flatppl_testsuite.unified.loader import load_test_module
from tests.test_interpolation import _gradient
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("dtype, tolerance", [("f64", 1e-11), ("f32", 2e-4)])
def test_composed_pyhf_helpers_preserve_scopes_and_dependencies(tmp_path, dtype, tolerance):
    _gate_engine("stablehlo")
    oracle = load_test_module(Path(__file__).parents[1] / "corpora/coverage/stdmod_interp_poly6")
    jax, _, _ = ex._jax()
    device = jax.devices()[0]
    query = tmp_path / "query.flatppl"
    query.write_text('pyhf = standard_module("pyhf_helpers", "0.1")\n'
                     "f(x) = cat(pyhf.normsys_factor.([0.8, 0.9], 1.2, x[[1, 1]]), "
                     "pyhf.normsys_factor.([0.7, 0.95, 1.1], [1.1], x[[3, 2, 3]]), "
                     "pyhf.normsys_factor.([0.8, 0.9], 1.3, "
                     "pyhf.normsys_factor.([0.9, 1.1], 1.2, x[2])))\n"
                     "g(y) = cat(pyhf.normsys_factor.([0.4, 0.9], 1.4, y[[2, 2]]), "
                     "pyhf.normsys_factor.([0.6, 0.8, 1.2], 1.1, y[[1, 1, 1]]))\n"
                     "points = elementof(cartpow(cartpow(reals, 3), 2))\n"
                     "inputs = points\noutputs = (f.(points), g.(points))\n")
    points = np.array([[-2., -.4, 1.], [.5, 2., -1.]],
                      dtype=np.float64 if dtype == "f64" else np.float32)
    def factor(lo, hi, alpha):
        return oracle.interp_poly6_exp(lo, 1., hi, alpha)
    expected_f = [[factor(l, 1.2, row[0]) for l in [.8, .9]]
                  + [factor(l, 1.1, row[i]) for l, i in zip([.7, .95, 1.1], [2, 1, 2])]
                  + [factor(l, 1.3, factor(inner, 1.2, row[1]))
                     for l, inner in zip([.8, .9], [.9, 1.1])] for row in points]
    expected_g = [[factor(l, 1.4, row[1]) for l in [.4, .9]]
                  + [factor(l, 1.1, row[0]) for l in [.6, .8, 1.2]] for row in points]
    fallback = tmp_path / "fallback.flatppl"
    subprocess.run([str(ex.flatppl_bin()), "determinize", str(query),
                    "--keep", "inputs", "--keep", "outputs", "-o", str(fallback)],
                   capture_output=True, text=True, check=True)
    for path in (query, fallback):
        executable = device.client.compile_and_load(ex.emit(path, "logdensity", dtype=dtype), [device])
        actual = executable.execute([jax.device_put(points)])
        for result, expected in zip(actual, [expected_f, expected_g]):
            np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
def test_integer_helper_anchors_keep_reference_promotions(tmp_path):
    _gate_engine("stablehlo")
    oracle = load_test_module(Path(__file__).parents[1] / "corpora/coverage/stdmod_interp_poly6")
    jax, _, _ = ex._jax()
    device = jax.devices()[0]
    query = tmp_path / "query.flatppl"
    for call, values, expected in [
        ("histosys_shift(lo, 1, hi, alpha)", (-9223372036854775807, 2, -2.), -2.**64),
        ("normsys_factor(lo, hi, alpha)", (9000000000000000000, 9000000000000000000, .5),
         oracle.interp_poly6_exp(9e18, 1., 9e18, .5)),
    ]:
        query.write_text('pyhf = standard_module("pyhf_helpers", "0.1")\n'
                         "lo = elementof(integers)\nhi = elementof(integers)\n"
                         "alpha = elementof(reals)\ninputs = (lo, hi, alpha)\n"
                         f"outputs = pyhf.{call}\n")
        fallback = tmp_path / "fallback.flatppl"
        subprocess.run([str(ex.flatppl_bin()), "determinize", str(query),
                        "--keep", "inputs", "--keep", "outputs", "-o", str(fallback)],
                       capture_output=True, text=True, check=True)
        args = [jax.device_put(np.array(v, dtype=t))
                for v, t in zip(values, (np.int64, np.int64, np.float64))]
        for path in (query, fallback):
            executable = device.client.compile_and_load(ex.emit(path, "logdensity", dtype="f64"), [device])
            np.testing.assert_allclose(executable.execute(args)[0], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("dtype, integer", [("f64", np.int64), ("f32", np.int32)])
def test_integer_nuisance_keeps_exponential_tail(tmp_path, dtype, integer):
    _gate_engine("stablehlo")
    jax, _, _ = ex._jax()
    device = jax.devices()[0]
    query = tmp_path / "query.flatppl"
    query.write_text('pyhf = standard_module("pyhf_helpers", "0.1")\n'
                     "alpha = elementof(integers)\ninputs = alpha\n"
                     "outputs = pyhf.normsys_factor(2.0, 1.0, alpha)\n")
    fallback = tmp_path / "fallback.flatppl"
    subprocess.run([str(ex.flatppl_bin()), "determinize", str(query),
                    "--keep", "inputs", "--keep", "outputs", "-o", str(fallback)],
                   capture_output=True, text=True, check=True)
    alpha = jax.device_put(np.array(np.iinfo(integer).min, dtype=integer))
    for path in (query, fallback):
        executable = device.client.compile_and_load(ex.emit(path, "logdensity", dtype=dtype), [device])
        # The negative tail is 2**(-alpha), which overflows positively.
        assert np.isposinf(np.asarray(executable.execute([alpha])[0])), path.name


@pytest.mark.stablehlo_only
def test_shared_pyhf_nuisances_preserve_selection_and_adjoint(tmp_path):
    _gate_engine("stablehlo")
    oracle = load_test_module(Path(__file__).parents[1] / "corpora/coverage/stdmod_interp_poly6")
    jax, jnp, hlo_call = ex._jax()
    points = np.array([[-2., -.4, 1., 17.], [0., 2., -1., -13.]])
    selected = np.array([2, 0, 2, 1, 0, 2])
    lo = np.array([.9, .2, 1.1, .8, .7, .99])
    hi = np.array([1.1, 3., .8, 1.2, 1.3, 1.01])
    weights = np.arange(1., 7.)
    query = tmp_path / "query.flatppl"
    for member, formula in [("normsys_factor", "interp_poly6_exp"),
                            ("histosys_shift", "interp_poly6_lin")]:
        anchors = f"{lo.tolist()}, {hi.tolist()}" if member == "normsys_factor" else (
            f"{lo.tolist()}, 1.0, {hi.tolist()}")
        query.write_text(
            'pyhf = standard_module("pyhf_helpers", "0.1")\n'
            f"evaluate(p) = pyhf.{member}.({anchors}, p[{(selected + 1).tolist()}])\n"
            "points = elementof(cartpow(cartpow(reals, 4), 2))\n"
            "inputs = points\noutputs = evaluate.(points)\n"
        )
        source = ex.emit(query, "logdensity", dtype="f64")
        evaluate = jax.jit(lambda xs: hlo_call(xs, source=source)[0])
        expected = np.array([[getattr(oracle, formula)(l, 1., h, row[index])
                              for l, h, index in zip(lo, hi, selected)] for row in points])
        if member == "histosys_shift":
            expected -= 1.
        expected_gradient = np.zeros_like(points)
        for row, point in enumerate(points):
            for l, h, index, weight in zip(lo, hi, selected, weights):
                alpha = point[index]
                if member == "normsys_factor":
                    slope = _gradient(oracle, l, 1., h, alpha)[-1]
                elif abs(alpha) >= 1:
                    slope = h - 1 if alpha > 0 else 1 - l
                else:
                    powers = np.arange(1, 7)
                    coefficients = oracle._poly6_coeffs(l, 1., h, 1 - l, h - 1, 0., 0.)
                    slope = np.sum(powers * coefficients * alpha ** (powers - 1))
                expected_gradient[row, index] += weight * slope
        gradient = jax.jit(jax.grad(lambda xs: jnp.sum(evaluate(xs) * weights)))
        np.testing.assert_allclose(evaluate(points), expected, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(gradient(points), expected_gradient, rtol=1e-11, atol=1e-11)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("dtype, tolerance", [("f64", 1e-11), ("f32", 2e-4)])
def test_normsys_reference_call_forms(tmp_path, dtype, tolerance):
    _gate_engine("stablehlo")
    oracle = load_test_module(Path(__file__).parents[1] / "corpora/coverage/stdmod_interp_poly6")
    (tmp_path / "reference.flatppl").write_text(
        'hep = standard_module("particle-physics", "0.1")\n'
        "normsys_factor(lo, hi, alpha) = hep.interp_poly6_exp(lo, 1.0, hi, alpha)\n"
    )
    points = np.array([[lo, hi, a] for lo, hi in [(0.9, 1.1), (0.2, 3.), (1., 1.)]
                      for a in [-2., -1., -0.999, 0., 0.999, 1., 2.]],
                     dtype=np.float64 if dtype == "f64" else np.float32)
    query = tmp_path / "query.flatppl"
    query.write_text(
        'pyhf = standard_module("pyhf_helpers", "0.1")\n'
        'reference = load_module("reference.flatppl")\n'
        "evaluate(p) = [pyhf.normsys_factor(p[1], p[2], p[3]),\n"
        "  pyhf.normsys_factor(p[2], alpha = -p[3], hi = p[1]),\n"
        "  pyhf.normsys_factor(record(hi = p[2], lo = p[1], alpha = p[3])),\n"
        "  reference.normsys_factor(p[1], hi = p[2], alpha = p[3])]\n"
        f"points = elementof(cartpow(cartpow(reals, 3), {len(points)}))\n"
        "inputs = points\noutputs = evaluate.(points)\n"
    )
    fallback = tmp_path / "fallback.flatppl"
    subprocess.run([str(ex.flatppl_bin()), "determinize", str(query),
                    "--keep", "inputs", "--keep", "outputs", "-o", str(fallback)],
                   capture_output=True, text=True, check=True)
    jax, jnp, hlo_call = ex._jax()
    expected = np.array([[oracle.interp_poly6_exp(lo, 1., hi, a),
                          oracle.interp_poly6_exp(hi, 1., lo, -a),
                          oracle.interp_poly6_exp(lo, 1., hi, a),
                          oracle.interp_poly6_exp(lo, 1., hi, a)]
                         for lo, hi, a in points])
    weights = np.arange(1, 5, dtype=points.dtype)
    expected_gradient = np.array([_gradient(oracle, lo, 1., hi, a)[[0, 2, 3]]
                                  for lo, hi, a in points]) * weights.sum()
    device = jax.devices()[0]
    for path in (query, fallback):
        source = ex.emit(path, "logdensity", dtype=dtype)
        executable = device.client.compile_and_load(source, [device])
        actual = executable.execute([jax.device_put(points)])[0]
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        source = ex.emit(path, "logdensity", dtype=dtype, restrict_enzyme_compatible=True)
        evaluate = jax.jit(lambda xs: hlo_call(xs, source=source)[0])
        gradient = jax.jit(jax.grad(lambda xs: jnp.sum(evaluate(xs) * weights)))
        np.testing.assert_allclose(evaluate(points), expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(gradient(points), expected_gradient, rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("dtype, tolerance", [("f64", 1e-11), ("f32", 2e-4)])
def test_normsys_adjoint_and_serialized_fallback(tmp_path, dtype, tolerance):
    _gate_engine("stablehlo")
    oracle = load_test_module(Path(__file__).parents[1] / "corpora/coverage/stdmod_interp_poly6")
    points = np.array([[lo, hi, a] for lo, hi in [(0.9, 1.1), (0.2, 3.), (1., 1.)]
                      for a in [-2., -1., -0.999, 0., 0.999, 1., 2.]],
                     dtype=np.float64 if dtype == "f64" else np.float32)
    query = tmp_path / "query.flatppl"
    query.write_text(
        'pyhf = standard_module("pyhf_helpers", "0.1")\n'
        "evaluate(p) = pyhf.normsys_factor(p[1], p[2], p[3])\n"
        f"points = elementof(cartpow(cartpow(reals, 3), {len(points)}))\n"
        "inputs = points\noutputs = evaluate.(points)\n"
    )
    fallback = tmp_path / "fallback.flatppl"
    subprocess.run([str(ex.flatppl_bin()), "determinize", str(query),
                    "--keep", "inputs", "--keep", "outputs", "-o", str(fallback)],
                   capture_output=True, text=True, check=True)
    jax, jnp, hlo_call = ex._jax()
    expected = np.array([oracle.interp_poly6_exp(lo, 1., hi, a) for lo, hi, a in points])
    weights = np.arange(1, len(points) + 1, dtype=points.dtype)
    expected_gradient = np.array([_gradient(oracle, lo, 1., hi, a)[[0, 2, 3]]
                                  for lo, hi, a in points]) * weights[:, None]
    for path in (query, fallback):
        source = ex.emit(path, "logdensity", dtype=dtype)
        evaluate = jax.jit(lambda xs: hlo_call(xs, source=source)[0])
        gradient = jax.jit(jax.grad(lambda xs: jnp.sum(evaluate(xs) * weights)))
        np.testing.assert_allclose(evaluate(points), expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(gradient(points), expected_gradient, rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
def test_normsys_inactive_tiny_anchor_tail_keeps_finite_adjoint(tmp_path):
    _gate_engine("stablehlo")
    jax, jnp, hlo_call = ex._jax()
    query = tmp_path / "query.flatppl"
    for module, call in [("pyhf_helpers", "normsys_factor(1e-320, 1.1, alpha)"),
                         ("particle-physics", "interp_poly6_exp(1e-320, 1.0, 1.1, alpha)")]:
        query.write_text(f'helper = standard_module("{module}", "0.1")\n'
                         "alpha = elementof(reals)\ninputs = alpha\n"
                         f"outputs = helper.{call}\n")
        source = ex.emit(query, "logdensity", dtype="f64")
        evaluate = jax.jit(lambda a: hlo_call(a, source=source)[0])
        np.testing.assert_allclose(evaluate(1.), 1.1, rtol=1e-12)
        np.testing.assert_allclose(jax.grad(evaluate)(1.), 1.1 * np.log(1.1), rtol=1e-12)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("dtype, tolerance", [("f64", 1e-11), ("f32", 2e-4)])
def test_histosys_zero_centered_shift_and_adjoint(tmp_path, dtype, tolerance):
    _gate_engine("stablehlo")
    oracle = load_test_module(Path(__file__).parents[1] / "corpora/coverage/stdmod_interp_poly6")
    points = np.array([[lo, nominal, hi, a]
                      for lo, nominal, hi in [(47., 50., 53.), (5., 10., 30.), (0., 0., 0.), (10., 0., 5.)]
                      for a in [-2., -1., -.999, 0., .999, 1., 2.]],
                     dtype=np.float64 if dtype == "f64" else np.float32)
    expected, derivatives = [], []
    powers = np.arange(1, 7)
    for lo, nominal, hi, a in points:
        expected.append(oracle.interp_poly6_lin(lo, nominal, hi, a) - nominal)
        anchors = [oracle.interp_poly6_lin(*basis, a) - basis[1]
                   for basis in [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]]
        coefficients = oracle._poly6_coeffs(lo, nominal, hi, nominal - lo, hi - nominal, 0., 0.)
        slope = (hi - nominal if a > 1 else nominal - lo if a < -1
                 else np.sum(powers * coefficients * a ** (powers - 1)))
        derivatives.append([*anchors, slope])
    query = tmp_path / "query.flatppl"
    query.write_text('pyhf = standard_module("pyhf_helpers", "0.1")\n'
                     "evaluate(p) = pyhf.histosys_shift(p[1], p[2], p[3], p[4])\n"
                     f"points = elementof(cartpow(cartpow(reals, 4), {len(points)}))\n"
                     "inputs = points\noutputs = evaluate.(points)\n")
    fallback = tmp_path / "fallback.flatppl"
    subprocess.run([str(ex.flatppl_bin()), "determinize", str(query),
                    "--keep", "inputs", "--keep", "outputs", "-o", str(fallback)],
                   capture_output=True, text=True, check=True)
    jax, jnp, hlo_call = ex._jax()
    weights = np.arange(1, len(points) + 1, dtype=points.dtype)
    for path in (query, fallback):
        source = ex.emit(path, "logdensity", dtype=dtype)
        evaluate = jax.jit(lambda xs: hlo_call(xs, source=source)[0])
        gradient = jax.jit(jax.grad(lambda xs: jnp.sum(evaluate(xs) * weights)))
        np.testing.assert_allclose(evaluate(points), expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(gradient(points), np.asarray(derivatives) * weights[:, None],
                                   rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
def test_histosys_preserves_small_shifts_and_negative_zero(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text('pyhf = standard_module("pyhf_helpers", "0.1")\n'
                     "evaluate(p) = pyhf.histosys_shift(p[1], p[2], p[3], p[4])\n"
                     "points = elementof(cartpow(cartpow(reals, 4), 2))\n"
                     "inputs = points\noutputs = evaluate.(points)\n")
    points = np.array([[.5, 1., 2., -0.], [1e30, 0., 1e30, 1e-30]], dtype=np.float32)
    jax, _, hlo_call = ex._jax()
    source = ex.emit(query, "logdensity", dtype="f32")
    evaluate = jax.jit(lambda xs: hlo_call(xs, source=source)[0])
    values = np.asarray(evaluate(points))
    assert values[0] == 0 and np.signbit(values[0])
    # For equal anchors and zero nominal, the leading term is 15/8 * anchor * alpha**2.
    # Squaring alpha in f32 first erases this representable shift.
    np.testing.assert_allclose(values[1], 1.875e-30, rtol=3e-6, atol=0)
    gradient = jax.jit(jax.grad(lambda xs: evaluate(xs)[1]))(points)
    np.testing.assert_allclose(gradient[1, 3], 3.75, rtol=3e-6, atol=0)


@pytest.mark.stablehlo_only
def test_fixed_histosys_anchors_preserve_broadcast_shape_and_adjoint(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text('pyhf = standard_module("pyhf_helpers", "0.1")\n'
                     "zero(a) = pyhf.histosys_shift.([3., 5.], [3., 5.], [3., 5.], a)\n"
                     "mixed(a) = pyhf.histosys_shift.([3., 4.], [3., 5.], [3., 6.], a)\n"
                     "points = elementof(cartpow(reals, 8))\n"
                     "inputs = points\noutputs = cat(zero.(points), mixed.(points))\n")
    jax, jnp, hlo_call = ex._jax()
    points = np.array([-2., -1., -.5, -0., 0., .5, 1., 2.])
    for restricted in (True, False):
        source = ex.emit(query, "logdensity", dtype="f64",
                         restrict_enzyme_compatible=restricted)
        evaluate = jax.jit(lambda xs: hlo_call(xs, source=source)[0])
        values = evaluate(points)
        zero, mixed = values[:8], values[8:]
        expected_zero = np.broadcast_to((points * 0.)[:, None], (8, 2))
        np.testing.assert_array_equal(zero, expected_zero)
        tails = np.abs(points) >= 1
        np.testing.assert_array_equal(np.signbit(zero[tails]), np.signbit(expected_zero[tails]))
        np.testing.assert_allclose(mixed, np.column_stack((np.zeros(8), points)), atol=1e-12)
        # Equal anchors give zero. Symmetric unit shifts give the identity.
        if restricted:
            gradient = jax.jit(jax.grad(lambda xs: jnp.sum(evaluate(xs))))
            np.testing.assert_allclose(gradient(points), np.ones(8), atol=1e-12)

"""Mixtures score atom masses and continuous densities in one fixed reference."""

import math

import numpy as np
import pytest
from scipy import special

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 2e-11)])
@pytest.mark.parametrize("law", ["Beta", "Gamma"])
@pytest.mark.parametrize("nested", [False, True])
def test_continuous_density_vanishes_at_fixed_atoms(tmp_path, precision, tolerance, law, nested):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "a = external(posreals)\nb = external(posreals)\n"
        "p = external(unitinterval)\nx = external(reals)\n"
        f"K(alpha, beta, w) = superpose(weighted(w, Dirac(0.0)), "
        f"weighted(1 - w, {law}(alpha, beta)))\n"
        + ("M = superpose(weighted(0.8, K(a, b, p)), weighted(0.2, Dirac(1.0)))\n"
           if nested else
           f"M = superpose(weighted(0.2, Dirac(1.0)), weighted(0.8 * (1 - p), {law}(a, b)), "
           "weighted(0.8 * p, Dirac(0.0)))\n")
        + "inputs = (a, b, p, x)\noutputs = logdensityof(M, x)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def score(point):
        return hlo_call(*point, source=source)[0]

    value_gradient = jax.jit(jax.value_and_grad(score))
    points = [[.2, .6, .3, 0], [.2, .6, .3, 1], [.2, .6, .3, .35],
              [.2, .6, 0, 1], [.2, .6, 1, 1]]
    for point in np.array(points, dtype=dtype):
        a, b, p, x = point.astype(np.float64)
        if x == 0:
            expected = math.log(.8 * p)
            gradient = [0, 0, 1 / p, 0]
        elif x == 1:
            expected = math.log(.2)
            gradient = [0, 0, 0, 0]
        elif law == "Beta":
            expected = (math.log(.8 * (1 - p)) + (a - 1) * math.log(x)
                        + (b - 1) * math.log1p(-x) - special.betaln(a, b))
            gradient = [math.log(x) - special.digamma(a) + special.digamma(a + b),
                        math.log1p(-x) - special.digamma(b) + special.digamma(a + b),
                        -1 / (1 - p), (a - 1) / x - (b - 1) / (1 - x)]
        else:
            expected = (math.log(.8 * (1 - p)) + a * math.log(b) - special.gammaln(a)
                        + (a - 1) * math.log(x) - b * x)
            gradient = [math.log(b * x) - special.digamma(a), a / b - x,
                        -1 / (1 - p), (a - 1) / x - b]
        actual, derivative = jax.block_until_ready(value_gradient(jnp.asarray(point)))
        assert actual.devices() == {jax.devices()[0]}
        assert derivative.devices() == {jax.devices()[0]}
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(derivative, gradient, rtol=tolerance, atol=tolerance)
        if x in (0, 1):
            np.testing.assert_array_equal(derivative[:2], [0, 0])


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_inactive_normalized_component_keeps_its_mass_precondition(tmp_path, precision):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "w = external(nonnegreals)\nx = external(reals)\n"
        "K(z) = normalize(weighted(z, Beta(0.2, 0.6)))\n"
        "M = superpose(weighted(0.8, K(w)), weighted(0.2, Dirac(0.0)))\n"
        "inputs = (w, x)\noutputs = logdensityof(M, x)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def score(weight):
        return hlo_call(weight, jnp.array(0, dtype=dtype), source=source)[0]

    compiled = jax.jit(jax.value_and_grad(score))
    score, gradient = jax.block_until_ready(compiled(jnp.array(1, dtype=dtype)))
    np.testing.assert_allclose(score, math.log(.2), rtol=1e-6)
    np.testing.assert_array_equal(gradient, 0)
    invalid, _ = jax.block_until_ready(compiled(jnp.array(0, dtype=dtype)))
    assert np.isnan(invalid)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_zero_weight_on_other_atom_has_finite_gradient(tmp_path, precision):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "p = external(unitinterval)\nx = external(reals)\n"
        "M = superpose(weighted(p, Dirac(0.0)), weighted(1 - p, Dirac(1.0)))\n"
        "inputs = (p, x)\noutputs = logdensityof(M, x)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def score(point):
        return hlo_call(*point, source=source)[0]

    compiled = jax.jit(jax.value_and_grad(score))
    for point, expected in [([0, 1], [-1, 0]), ([1, 0], [1, 0])]:
        score, gradient = jax.block_until_ready(compiled(jnp.array(point, dtype=dtype)))
        np.testing.assert_array_equal(score, 0)
        np.testing.assert_array_equal(gradient, expected)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 2e-11)])
@pytest.mark.parametrize("model,mass,slope", [
    ("superpose(weighted(p, A), weighted(1 - p, A))", 1, 0),
    ("superpose(weighted(2 * p, A), superpose(weighted(3 * (1 - p), A), "
     "weighted(0.4, Dirac(1.0))))", 3, -1),
    ("K(p)", 3, -1),
])
def test_coincident_atoms_keep_endpoint_weight_gradients(
    tmp_path, precision, tolerance, model, mass, slope,
):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "p = external(unitinterval)\nx = external(reals)\n"
        "D = Dirac(0.0)\nA = D\n"
        "K(p) = superpose(weighted(2 * p, A), weighted(3 * (1 - p), A))\n"
        f"M = {model}\ninputs = (p, x)\noutputs = logdensityof(M, x)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64

    def score(p):
        return hlo_call(p, jnp.array(0, dtype=dtype), source=source)[0]

    points = jnp.array([0, .25, .5, 1], dtype=dtype)
    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(score)))(points)
    expected_mass = mass + slope * np.asarray(points)
    np.testing.assert_allclose(values, np.log(expected_mass), rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(gradients, slope / expected_mass, rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,weight", [("f32", 3e38), ("f64", 1e308)])
def test_coincident_atom_log_mass_survives_linear_sum_overflow(tmp_path, precision, weight):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "w = external(nonnegreals)\n"
        "M = superpose(weighted(w, Dirac(0.0)), weighted(w, Dirac(0.0)))\n"
        "inputs = (w)\noutputs = logdensityof(M, 0.0)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    np.testing.assert_allclose(ex.value(source, [weight]), math.log(weight) + math.log(2), rtol=1e-6)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_coincident_atom_mass_guards(tmp_path, precision):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "p = external(reals)\nx = external(reals)\n"
        "M = superpose(weighted(p, Dirac(0.0)), weighted(2 * p, Dirac(0.0)), Dirac(1.0))\n"
        "inputs = (p, x)\noutputs = logdensityof(M, x)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    assert ex.value(source, [0, 1]) == 0
    np.testing.assert_array_equal(ex.gradient(source, [0, 1], argnums=(0, 1)), [0, 0])
    assert ex.value(source, [0, 0]) == -math.inf
    assert ex.value(source, [math.inf, 0]) == math.inf
    assert np.isnan(ex.value(source, [-1, 0]))
    assert np.isnan(ex.value(source, [math.nan, 0]))

    query.write_text(
        "p = external(unitinterval)\nx = external(reals)\n"
        "M = superpose(superpose(weighted(p, Dirac(0.0)), Dirac(1.0)), "
        "superpose(weighted(1 - p, Dirac(0.0)), Dirac(2.0)))\n"
        "inputs = (p, x)\noutputs = logdensityof(M, x)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    assert ex.value(source, [0, 3]) == -math.inf


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,atom", [("f32", 16777217), ("f64", 9007199254740992)])
def test_coincident_atoms_keep_integer_support(tmp_path, precision, atom):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "p = external(unitinterval)\nx = external(integers)\n"
        f"M = superpose(weighted(p, Dirac({atom})), weighted(1 - p, Dirac({atom})), "
        f"Dirac({atom + 1}))\ninputs = (p, x)\noutputs = logdensityof(M, x)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    for point in [[0, atom], [1, atom], [.5, atom + 1]]:
        assert ex.value(source, point) == 0
        np.testing.assert_array_equal(ex.gradient(source, point, argnums=(0,)), [0])

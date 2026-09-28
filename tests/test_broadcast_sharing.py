"""Shared cell arrays retain their scope, batch domain, and derivatives."""

import subprocess

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("count", [1, 4])
def test_shared_cell_arrays_use_runtime_points(tmp_path, count):
    _gate_engine("stablehlo")
    (tmp_path / "model.flatppl").write_text(
        "x = elementof(reals)\n"
        "rates = exp.([x, x + 1.0, x + 2.0])\n"
        "L = likelihoodof(Normal(sum(rates) + sum(rates .* rates), 1.0), 0.2)\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        'm = load_module("model.flatppl")\n'
        "score(t) = logdensityof(m.L, record(x = t))\n"
        f"points = elementof(cartpow(reals, {count}))\n"
        "inputs = points\noutputs = sum(score.(points))\n"
    )
    serialized = tmp_path / "lowered.flatppl"
    subprocess.run(
        [str(ex.flatppl_bin()), "determinize", str(query), "--keep", "inputs",
         "--keep", "outputs", "-o", str(serialized)],
        check=True, capture_output=True, text=True,
    )
    source = ex.emit(serialized, "logdensity", dtype="f64")
    points = np.linspace(-1.0, 0.5, count)
    rates = np.exp(points[:, None] + np.arange(3))
    mean = rates.sum(axis=1) + (rates * rates).sum(axis=1)
    expected = (-0.5 * (np.log(2 * np.pi) + (mean - 0.2)**2)).sum()
    gradient = -(mean - 0.2) * (rates.sum(axis=1) + 2 * (rates * rates).sum(axis=1))
    np.testing.assert_allclose(ex.value(source, [points]), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ex.gradient(source, [points], [0]), [gradient], rtol=1e-12, atol=1e-12)


@pytest.mark.stablehlo_only
def test_short_selections_keep_shared_children_and_weighted_adjoint(tmp_path):
    _gate_engine("stablehlo")
    (tmp_path / "model.flatppl").write_text(
        "score(t) = sum(2.0 * [t[12]] * exp(t[1]) * exp(t[3])\n"
        "  + 3.0 * [t[12]] * exp(t[3]) * exp(t[8])\n"
        "  + 4.0 * [t[12]] * exp(t[7]) * exp(t[11])) + sin(exp(t[7]))\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        'm = load_module("model.flatppl")\nscore = m.score\n'
        "points = elementof(cartpow(cartpow(reals, 12), 3))\n"
        "inputs = points\noutputs = score.(points)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, hlo_call = ex._jax()
    evaluate = jax.jit(lambda xs: hlo_call(xs, source=source)[0])
    points = np.arange(36, dtype=float).reshape(3, 12) / 32 - 0.5
    left, right = [0, 2, 6], [2, 7, 10]
    factors = np.array([2, 3, 4]) * np.exp(points[:, left]) * np.exp(points[:, right])
    terms = points[:, 11, None] * factors
    shared = np.exp(points[:, 6])
    expected = terms.sum(axis=1) + np.sin(shared)
    np.testing.assert_allclose(evaluate(points), expected, rtol=1e-12, atol=1e-12)
    weights = np.array([1.0, -2.0, 3.0])
    gradient = jax.jit(jax.grad(lambda xs: jnp.sum(evaluate(xs) * weights)))
    expected_gradient = np.zeros_like(points)
    for i, (a, b) in enumerate(zip(left, right)):
        expected_gradient[:, a] += terms[:, i]
        expected_gradient[:, b] += terms[:, i]
    expected_gradient[:, 11] = factors.sum(axis=1)
    expected_gradient[:, 6] += np.cos(shared) * shared
    np.testing.assert_allclose(
        gradient(points), expected_gradient * weights[:, None], rtol=1e-12, atol=1e-12,
    )

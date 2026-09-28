"""Interpolation values and gradients obey the C² matching conditions."""

from pathlib import Path

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from flatppl_testsuite.unified.loader import load_test_module
from tests.test_unified import _gate_engine


def _gradient(oracle, left, center, right, alpha):
    lo, hi = np.log(left / center), np.log(right / center)
    value = oracle.interp_poly6_exp(left, center, right, alpha)
    if alpha > 1:
        return np.array([0, (1 - alpha) * value / center, alpha * value / right, value * hi])
    if alpha < -1:
        return np.array([-alpha * value / left, (1 + alpha) * value / center, 0, -value * lo])
    # Differentiate the independent C² linear system, not the lowered polynomial.
    solve = oracle._poly6_coeffs
    coefficients = solve(left, center, right, -left * lo, right * hi, left * lo**2, right * hi**2)
    powers = np.arange(1, 7)
    derivatives = [
        solve(1, 0, 0, -lo - 1, 0, lo**2 + 2 * lo, 0),
        solve(0, 1, 0, left / center, -right / center, -2 * left * lo / center, -2 * right * hi / center),
        solve(0, 0, 1, 0, hi + 1, 0, hi**2 + 2 * hi),
    ]
    anchors = np.array(derivatives) @ (alpha**powers) + [0, 1, 0]
    return np.append(anchors, np.sum(powers * coefficients * alpha ** (powers - 1)))


@pytest.mark.stablehlo_only
def test_exponential_interpolation_dynamic_anchors_and_adjoint(tmp_path):
    _gate_engine("stablehlo")
    oracle = load_test_module(Path(__file__).parents[1] / "corpora/coverage/stdmod_interp_poly6")
    points = np.array(
        [[0.8, 2.0, 3.0, a] for a in [-2, -1, -0.4, 0, 0.6, 1, 2]]
        # Finite active tails whose discarded opposite exponentials overflow.
        + [[0.1, 1.0, 1.1, 400], [1.1, 1.0, 0.1, -400]]
    )
    (tmp_path / "model.flatppl").write_text(
        'flatppl_compat = "0.1"\nhep = standard_module("particle-physics", "0.1")\n'
        "interpolate(p) = hep.interp_poly6_exp(p[1], p[2], p[3], p[4])\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        'flatppl_compat = "0.1"\nm = load_module("model.flatppl")\ninterpolate = m.interpolate\n'
        f"points = elementof(cartpow(cartpow(reals, 4), {len(points)}))\n"
        "inputs = points\noutputs = interpolate.(points)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, hlo_call = ex._jax()
    evaluate = jax.jit(lambda xs: hlo_call(xs, source=source)[0])
    expected = np.array([oracle.interp_poly6_exp(*point) for point in points])
    np.testing.assert_allclose(evaluate(points), expected, rtol=1e-12, atol=1e-12)
    weights = np.arange(1, len(points) + 1) * (-1.0) ** np.arange(len(points))
    gradient = jax.jit(jax.grad(lambda xs: jnp.sum(evaluate(xs) * weights)))
    expected_gradient = np.array([_gradient(oracle, *point) for point in points]) * weights[:, None]
    np.testing.assert_allclose(gradient(points), expected_gradient, rtol=1e-11, atol=1e-12)

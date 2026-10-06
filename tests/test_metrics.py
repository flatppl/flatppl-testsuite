"""Metric contractions preserve variance, tensor axes, batches and adjoints."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
def test_runtime_metric_pivots_batches_and_gradients(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "gs = elementof(cartpow(cartpow(reals, [2,2]), 2))\n"
        "v = elementof(cartpow(reals, 2))\n"
        "f(g) = metricsum(g, [], v[.i^]*v[.i_])\n"
        "inputs = (gs, v)\noutputs = f.(gs)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, hlo_call = ex._jax()
    evaluate = jax.jit(lambda gs, v: hlo_call(gs, v, source=source)[0])
    metrics = np.array([[[0., 2.], [2., 3.]], [[4., 1.], [1., 2.]]])
    point = np.array([1., 2.])
    solved = np.linalg.solve(metrics, np.broadcast_to(point, (2, 2))[..., None])[..., 0]
    np.testing.assert_allclose(evaluate(metrics, point), np.einsum("i,bi->b", point, solved), atol=1e-12)
    gradient = jax.jit(jax.grad(lambda gs, v: jnp.sum(evaluate(gs, v)), argnums=(0, 1)))
    metric_grad, point_grad = gradient(metrics, point)
    np.testing.assert_allclose(metric_grad, -np.einsum("bi,bj->bij", solved, solved), atol=1e-12)
    np.testing.assert_allclose(point_grad, 2*solved.sum(axis=0), atol=1e-12)


@pytest.mark.stablehlo_only
def test_metric_repeated_selectors_and_reordered_output(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "g = elementof(cartpow(reals, [2,2]))\n"
        "T = elementof(cartpow(reals, [2,2,2]))\n"
        "g: result[.k_, .i_] := T[.i_, .i_, .k_]\n"
        "inputs = (g, T)\noutputs = result\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, hlo_call = ex._jax()
    evaluate = jax.jit(lambda g, t: hlo_call(g, t, source=source)[0])
    metric = np.array([[0., 2.], [2., 3.]])
    tensor = np.arange(8, dtype=float).reshape(2, 2, 2)/3 - 1

    def oracle(g, t):
        inverse = jnp.linalg.inv(g)
        lower = jnp.einsum("ia,ib,kc,abc->ki", inverse, inverse, inverse, t)
        return g @ lower @ g.T

    np.testing.assert_allclose(evaluate(metric, tensor), oracle(metric, tensor), atol=1e-12)
    weights = jnp.array([[1., -2.], [3., .5]])
    actual = jax.jit(jax.grad(lambda g, t: jnp.sum(weights*evaluate(g, t)), argnums=(0, 1)))(metric, tensor)
    expected = jax.grad(lambda g, t: jnp.sum(weights*oracle(g, t)), argnums=(0, 1))(metric, tensor)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=1e-11)


@pytest.mark.stablehlo_only
def test_metric_callback_reuse_and_promoted_fixed_input(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "g = rowstack([[2.,0.],[0.,-4.]])\n"
        "h = rowstack([[3.,0.],[0.,5.]])\n"
        "v = elementof(cartpow(reals,2))\n"
        "f(m) = metricsum(m, [.i_], 2*v[.i_])\n"
        "g: norm[] := v[.i^]*v[.i_]\n"
        "inputs = (g, v)\noutputs = sum(f(g)) + 3*sum(f(h)) + norm\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    metric = np.array([[0., 2.], [2., 3.]])
    point = np.array([1., 2.])
    expected = 8*point.sum() + point @ np.linalg.solve(metric, point)
    np.testing.assert_allclose(ex.value(source, [metric, point]), expected, atol=1e-12)

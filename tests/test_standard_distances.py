"""Standard distances execute with vector cells, callbacks, and zero masses."""

import numpy as np
import pytest
from scipy.spatial import distance

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-6), ("f64", 2e-11)])
def test_distances_preserve_vector_cells_and_callback_captures(
    tmp_path, precision, tolerance,
):
    _gate_engine("stablehlo")
    query = tmp_path / "distances.flatppl"
    query.write_text(
        'd = standard_module("distances", "0.1")\n'
        "u = external(cartpow(reals, 3))\n"
        "v = external(cartpow(reals, 3))\n"
        "p = external(posreals)\n"
        "x = external(cartpow(cartpow(reals, 2), 3))\n"
        "y = external(cartpow(cartpow(reals, 2), 2))\n"
        "metric(a, b) = sum(a) + 2 * sum(b) + p\n"
        "inputs = (u, v, p, x, y)\n"
        "outputs = (d.minkowski(u, v, p), d.jensenshannon(u, v), "
        "d.pairwise_distance(d.euclidean, x), "
        "d.cross_distance(d.euclidean, x, y), "
        "d.cross_distance(metric, x, y))\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    call = ex.call_with_status(source)
    evaluate = jax.jit(call)
    u = np.array([0, .25, .75], dtype=dtype)
    v = np.array([.5, .5, 0], dtype=dtype)
    p = np.array(3.7, dtype=dtype)
    x = np.array([[0, 0], [1, 0], [1, 1]], dtype=dtype)
    y = np.array([[2, 0], [2, 3]], dtype=dtype)
    args = tuple(jnp.asarray(a) for a in (u, v, p, x, y))
    actual = ex.expect_valid(evaluate(*args))
    expected = (
        distance.minkowski(u, v, float(p)),
        distance.jensenshannon(u, v),
        distance.cdist(x, x),
        distance.cdist(x, y),
        x.sum(axis=1)[:, None] + 2 * y.sum(axis=1)[None, :] + p,
    )
    for got, wanted in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got, wanted, rtol=tolerance, atol=tolerance)

    # Differentiate the exponent at runtime. Its analytic derivative follows
    # from log(norm) = log(sum(abs(u-v)**p)) / p.
    def norm(u, v, p, x, y):
        return call(u, v, p, x, y)[0][0]

    delta = np.abs(u.astype(float) - v.astype(float))
    powered = delta ** float(p)
    total = powered.sum()
    derivative = expected[0] * (
        np.dot(powered, np.log(delta)) / (float(p) * total)
        - np.log(total) / float(p) ** 2
    )
    actual_derivative = jax.jit(jax.grad(norm, argnums=2))(*args)
    np.testing.assert_allclose(
        actual_derivative, derivative, rtol=5 * tolerance, atol=tolerance,
    )

    def callback_sum(u, v, p, x, y):
        return jnp.sum(call(u, v, p, x, y)[0][4])

    actual_derivative = jax.jit(jax.grad(callback_sum, argnums=2))(*args)
    np.testing.assert_allclose(actual_derivative, 6, rtol=tolerance)


@pytest.mark.stablehlo_only
def test_jensen_shannon_retains_small_distances(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "small-distance.flatppl"
    query.write_text(
        'd = standard_module("distances", "0.1")\n'
        "a = external(cartpow(reals, 2))\n"
        "b = external(cartpow(reals, 2))\n"
        "inputs = (a, b)\noutputs = d.jensenshannon(a, b)\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    assert ex.value(source, [[1., 0.], [1., 0.]]) == 0
    assert ex.value(source, [[1., 0.], [0., 1.]]) == pytest.approx(np.sqrt(np.log(2)))
    for delta in (1e-6, 1e-8, 1e-10):
        a = np.array([.5 + delta, .5 - delta])
        b = a[::-1].copy()
        # At symmetric points, JSD² = t²/2 + t⁴/12 + t⁶/30 + O(t⁸),
        # where t is the represented difference a[0]-a[1].
        t = float(a[0] - a[1])
        expected = abs(t) * np.sqrt(.5 + t * t / 12 + t ** 4 / 30)
        assert ex.value(source, [a, b]) == pytest.approx(expected, rel=2e-8, abs=0)

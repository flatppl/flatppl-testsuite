"""Mixture draws must follow the weighted sum of their components."""

import numpy as np
import pytest
from scipy import stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

N = 20000
SEED = 0x123456789ABCDEF


def _sampler(tmp_path, source, precision):
    query = tmp_path / "query.flatppl"
    query.write_text(source)
    module = ex.emit(query, "sample", dtype=precision, restrict_enzyme_compatible=False)
    jax, _, _ = ex._jax()
    draw = jax.jit(ex.call_with_status(module))
    return lambda *args: ex.expect_valid(draw(*args))


def _integer(precision):
    return np.int32 if precision == "f32" else np.int64


def _draw(tmp_path, source, precision, *inputs, integer=False):
    _, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    draw = _sampler(tmp_path, source, precision)
    key = jnp.asarray(ex.rng_state(SEED))
    values = draw(key, *[jnp.asarray(x, dtype=dtype) for x in inputs])[0]
    # Section 08: a discrete variate is an integer, at the dtype's width.
    assert values.dtype == (_integer(precision) if integer else dtype)
    return np.asarray(values, dtype=np.float64)


def _normal_mixture_cdf(weights, means, sigmas):
    weights = np.asarray(weights, dtype=float) / np.sum(weights)
    return lambda x: sum(
        w * stats.norm.cdf(x, mu, sigma) for w, mu, sigma in zip(weights, means, sigmas)
    )


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_component_frequencies_follow_the_weights(tmp_path, precision):
    _gate_engine("stablehlo")
    values = _draw(tmp_path,
        "w = [0.2, 0.3, 0.5]\n"
        "M = normalize(ksuperpose(Dirac, w, value = [1.0, 2.0, 3.0]))\n"
        f"outputs = rand(iid(M, {N}), rnginit(0))\n", precision)
    counts = np.array([(values == label).sum() for label in (1, 2, 3)])
    assert counts.sum() == N
    assert stats.chisquare(counts, N * np.array([0.2, 0.3, 0.5])).pvalue > 1e-3


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize(
    "mixture, weights, means, sigmas",
    [
        ("ksuperpose(Normal, [1.0, 3.0], mu = [-3.0, 3.0], sigma = 1.0)",
         [1, 3], [-3, 3], [1, 1]),
        ("ksuperpose(Normal, [1.0, 2.0], table(mu = [-3.0, 3.0], sigma = [0.5, 2.0]))",
         [1, 2], [-3, 3], [0.5, 2]),
        ("ksuperpose(Normal, [1.0, 2.0], [-3.0, 3.0], sigma = [0.5, 2.0])",
         [1, 2], [-3, 3], [0.5, 2]),
        ("ksuperpose(Normal, [2.0], mu = [-3.0, 0.0, 3.0], sigma = [1.0])",
         [1, 1, 1], [-3, 0, 3], [1, 1, 1]),
    ],
    ids=["named", "table", "mixed", "singleton"],
)
def test_normal_mixture_draws_follow_the_mixture_cdf(
    tmp_path, precision, mixture, weights, means, sigmas
):
    _gate_engine("stablehlo")
    values = _draw(tmp_path,
        f"outputs = rand(iid(normalize({mixture}), {N}), rnginit(0))\n", precision)
    cdf = _normal_mixture_cdf(weights, means, sigmas)
    assert stats.kstest(values, cdf).pvalue > 1e-3


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_multiple_axes_expand_each_weight_over_its_broadcast_cells(tmp_path, precision):
    _gate_engine("stablehlo")
    # w is [2, 1] and mu is [2, 3]: each weight covers three cells.
    weights = np.array([[1.0], [3.0]])
    means = 10.0 * np.arange(6.0).reshape(2, 3)
    values = _draw(tmp_path,
        "w = external(cartpow(nonnegreals, [2, 1]))\n"
        "mu = external(cartpow(reals, [2, 3]))\ninputs = (w, mu)\n"
        "M = normalize(ksuperpose(Normal, w, mu, 0.001))\n"
        f"outputs = rand(iid(M, {N}), rnginit(0))\n", precision, weights, means)
    cells = np.rint(values / 10).astype(int)
    counts = np.bincount(cells, minlength=6)
    expected = N * np.repeat(weights.ravel(), 3) / (3 * weights.sum())
    assert stats.chisquare(counts, expected).pvalue > 1e-3


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_poisson_mixture_counts_follow_the_mixture_pmf(tmp_path, precision):
    _gate_engine("stablehlo")
    values = _draw(tmp_path,
        "M = normalize(ksuperpose(Poisson, [1.0, 3.0], rate = [1.0, 8.0]))\n"
        f"outputs = rand(iid(M, {N}), rnginit(0))\n", precision, integer=True)
    support = np.arange(21)
    pmf = 0.25 * stats.poisson.pmf(support, 1) + 0.75 * stats.poisson.pmf(support, 8)
    pmf[-1] += 1 - pmf.sum()
    counts = np.bincount(np.minimum(values.astype(int), 20), minlength=21)
    assert stats.chisquare(counts, N * pmf).pvalue > 1e-3


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("lanes", ["", ", 8"], ids=["single", "iid"])
def test_invalid_runtime_mass_reports_a_status(tmp_path, precision, lanes):
    _gate_engine("stablehlo")
    _, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    law = "normalize(ksuperpose(Normal, w, mu = [-3.0, 3.0], sigma = 1.0))"
    law = f"iid({law}{lanes})" if lanes else law
    query = tmp_path / "query.flatppl"
    query.write_text(
        "w = external(cartpow(reals, 2))\ninputs = w\n"
        f"outputs = rand({law}, rnginit(0))\n"
    )
    module = ex.emit(query, "sample", dtype=precision, restrict_enzyme_compatible=False)
    assert ex._runtime_status_index(module) is not None
    jax, _, _ = ex._jax()
    draw = jax.jit(ex.call_with_status(module))
    key = jnp.asarray(ex.rng_state(SEED))
    # Zero mass, a negative weight, infinite mass and a NaN weight leave the
    # normalized weights outside the simplex: status 7, not a draw.
    for weights in ([0.0, 0.0], [-1.0, 2.0], [1.0, np.inf], [np.nan, 1.0]):
        _, status = draw(key, jnp.asarray(weights, dtype=dtype))
        assert int(status) == 7, weights
    result = ex.expect_valid(draw(key, jnp.asarray([0.0, 1.0], dtype=dtype)))
    # A zero weight never selects its component.
    assert np.all(np.asarray(result[0]) > 0)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_mixture_draws_thread_the_rng(tmp_path, precision):
    _gate_engine("stablehlo")
    _, jnp, _ = ex._jax()
    mixture = "normalize(ksuperpose(Normal, [1.0, 3.0], mu = [-3.0, 3.0], sigma = 1.0))"
    single = _sampler(tmp_path, f"outputs = rand({mixture}, rnginit(0))\n", precision)
    chain = _sampler(tmp_path,
        f"x, state = rand({mixture}, rnginit(0))\n"
        f"y, advanced = rand({mixture}, state)\n"
        "outputs = (x, y, advanced)\n", precision)
    key = jnp.asarray(ex.rng_state(SEED))
    value, state = single(key)[:2]
    next_value, advanced = single(state)[:2]
    together = chain(key)
    np.testing.assert_array_equal(value, together[0])
    np.testing.assert_array_equal(next_value, together[1])
    np.testing.assert_array_equal(advanced, together[2])
    assert not np.array_equal(key, state)

"""Execute declared ABI types without silently changing input values."""

from pathlib import Path

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("product", ["callable", "iid", "process"])
@pytest.mark.parametrize("mixture", [
    "superpose(weighted(0.4, Normal(mu, 1.0)), weighted(0.6, Normal(3.0, 2.0)))",
    "ksuperpose(Normal, [0.4, 0.6])(mu = [mu, 3.0], sigma = [1.0, 2.0])",
])
def test_mixture_broadcast_preserves_observation_axis(tmp_path, mixture, product):
    _gate_engine("stablehlo")
    model = tmp_path / "model.flatppl"
    score = {
        "callable": "sum(fn(logdensityof(M, _)).(x))",
        "iid": "logdensityof(iid(M, 20), x)",
        "process": "logdensityof(PoissonProcess(M), x)",
    }[product]
    model.write_text(
        "mu = elementof(reals)\n"
        f"M = {mixture}\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        "mu = elementof(reals)\nx = elementof(cartpow(reals, 20))\n"
        'm = load_module("model.flatppl", mu = mu)\n'
        f"M = m.M\nscore = {score}\n"
        "inputs = (mu, x)\noutputs = (score)\n"
    )
    src = ex.emit(query, "logdensity")
    x = np.linspace(-2, 6, 20)
    a = np.log(0.4) - 0.5 * np.log(2 * np.pi) - 0.5 * x**2
    b = np.log(0.6) - np.log(2) - 0.5 * np.log(2 * np.pi) - 0.5 * ((x - 3) / 2)**2
    logp = np.logaddexp(a, b)
    expected = logp.sum() - (1.0 if product == "process" else 0.0)
    assert ex.value(src, [0.0, x]) == pytest.approx(expected, rel=0, abs=1e-5)
    # d log p(x) / d mu = responsibility of component 1 times (x - mu).
    derivative = np.sum(np.exp(a - logp) * x)
    assert ex.gradient(src, [0.0, x], [0]) == pytest.approx([derivative], rel=0, abs=1e-5)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("application", [
    "K.(weights, 2.0)", "K.(b = 2.0, a = weights)",
    "K.(weights, b = 2.0)", "R.(table(a = weights, b = [2.0, 2.0, 2.0]))",
])
def test_composed_kernel_broadcast_preserves_arguments_and_captures(tmp_path, application):
    _gate_engine("stablehlo")
    (tmp_path / "model.flatppl").write_text(
        "mu = elementof(reals)\n"
        "K(a, b) = normalize(superpose(weighted(a, Normal(mu, 1.0)), "
        "weighted(b, Normal(3.0, 2.0))))\n"
        "R(row) = K(row.a, row.b)\n"
        "weights = [0.5, 2.0, 4.0]\n"
        f"M = {application}\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        "mu = elementof(reals)\nx = elementof(cartpow(reals, 3))\n"
        'm = load_module("model.flatppl", mu = mu)\n'
        "inputs = (mu, x)\noutputs = logdensityof(m.M, x)\n"
    )
    src = ex.emit(query, "logdensity", dtype="f64")
    x = np.array([-0.2, 0.7, 4.0])
    weights = np.array([0.5, 2.0, 4.0])
    for mu in (-0.7, 1.2):
        a = np.log(weights) - 0.5 * np.log(2 * np.pi) - 0.5 * (x - mu)**2
        b = -0.5 * np.log(2 * np.pi) - 0.5 * ((x - 3) / 2)**2
        logp = np.logaddexp(a, b)
        expected = (logp - np.log(weights + 2)).sum()
        derivative = np.sum(np.exp(a - logp) * (x - mu))
        np.testing.assert_allclose(ex.value(src, [mu, x]), expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(ex.gradient(src, [mu, x], [0]), [derivative], rtol=0, atol=1e-12)


@pytest.mark.stablehlo_only
def test_scalar_kernel_broadcast_preserves_vector_variate(tmp_path):
    _gate_engine("stablehlo")
    (tmp_path / "model.flatppl").write_text(
        "mu = elementof(reals)\nK = functionof(iid(Normal(mu, 1.0), 3))\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        'm = load_module("model.flatppl")\n'
        "theta = elementof(reals)\nx = elementof(cartpow(reals, 3))\n"
        "K = m.K\ninputs = (theta, x)\noutputs = logdensityof(K.(mu = theta), x)\n"
    )
    src = ex.emit(query, "logdensity", dtype="f64")
    x = np.array([-0.2, 0.7, 4.0])
    mu = 1.2
    expected = -1.5 * np.log(2 * np.pi) - 0.5 * np.sum((x - mu)**2)
    np.testing.assert_allclose(ex.value(src, [mu, x]), expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ex.gradient(src, [mu, x], [0]), [np.sum(x - mu)], rtol=0, atol=1e-12)


@pytest.mark.stablehlo_only
def test_composed_kernel_broadcast_code_size_does_not_scale_with_observations(tmp_path):
    _gate_engine("stablehlo")
    (tmp_path / "model.flatppl").write_text(
        "K(mu) = superpose(weighted(0.4, Normal(mu, 1.0)), "
        "weighted(0.6, Normal(3.0, 2.0)))\n"
    )
    query = tmp_path / "query.flatppl"
    operation_counts = []
    for n in (3, 20000):
        query.write_text(
            'm = load_module("model.flatppl")\nK = m.K\n'
            f"mu = elementof(cartpow(reals, {n}))\nx = elementof(cartpow(reals, {n}))\n"
            "inputs = (mu, x)\noutputs = logdensityof(K.(mu), x)\n"
        )
        src = ex.emit(query, "logdensity", dtype="f64")
        operation_counts.append(sum(" = " in line for line in src.splitlines()))
    assert operation_counts[0] == operation_counts[1]


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_emitter_folds_constant_math_without_freezing_runtime_inputs(tmp_path, dtype):
    _gate_engine("stablehlo")
    model = tmp_path / "model.flatppl"
    model.write_text(
        "counts = [0, 3, 7]\n"
        "term(count, p) = loggamma(count + 1) + p\n"
        "score(p) = sum(term.(counts, [p]))\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        "points = elementof(cartpow(reals, 2))\n"
        'm = load_module("model.flatppl")\n'
        "score = m.score\ninputs = (points)\noutputs = sum(score.(points))\n"
    )
    src = ex.emit(query, "logdensity", dtype=dtype)
    assert "chlo.lgamma" not in src
    # Gamma(n + 1) = n!, independently of the folding library.
    points = [7.5, 2.25]
    expected = 2 * np.log([1.0, 6.0, 5040.0]).sum() + 3 * sum(points)
    np.testing.assert_allclose(ex.value(src, [points]), expected, rtol=0,
                               atol=1e-5 if dtype == "f32" else 1e-12)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("law, observations, expected, gradient_supported", [
    ("MvNormal([mu, 0.0], [[1.0, 0.0], [0.0, 1.0]])",
     "[[0.2, 0.3], [0.4, -0.1], [0.0, 0.5]]",
     -3 * np.log(2 * np.pi) - 0.5 * 0.55, False),
    ("joint(a = Normal(mu, 1.0), b = Normal(0.0, 1.0))",
     "table(a = [0.2, 0.4, 0.0], b = [0.3, -0.1, 0.5])",
     -3 * np.log(2 * np.pi) - 0.5 * 0.55, True),
])
def test_iid_broadcast_preserves_multivariate_cells(
    tmp_path, law, observations, expected, gradient_supported,
):
    _gate_engine("stablehlo")
    model = tmp_path / "model.flatppl"
    model.write_text(
        "mu = elementof(reals)\n"
        f"M = iid({law}, 3)\nobservations = {observations}\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        "mu = elementof(reals)\n"
        'm = load_module("model.flatppl", mu = mu)\n'
        "score = logdensityof(m.M, m.observations)\n"
        "inputs = (mu)\noutputs = (score)\n"
    )
    src = ex.emit(query, "logdensity")
    for mu in (0.0, 0.7):
        want = expected + 0.6 * mu - 1.5 * mu**2
        assert ex.value(src, [mu]) == pytest.approx(want, rel=0, abs=1e-5)
    # The pinned Enzyme backend has no triangular_solve adjoint (also on base).
    # Keep MvNormal values covered; the table case checks the supported gradient.
    if gradient_supported:
        assert ex.gradient(src, [0.7], [0]) == pytest.approx([-1.5], rel=0, abs=1e-5)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize(("body", "expected"), [
    ("x = elementof(reals)\ng = functionof(x + 1.0)\nf = functionof(g(x = x))\nscore = f(x = sum(xs)) + f(x = 2.0)", 8.5),
    ("f(x) = x\nscore = sum(f(xs)) + f(xs[1]) + sum(f([xs[2], xs[3]]))", 9.0),
    ('hep = standard_module("particle-physics", "0.1")\ng = hep.kallen\nscore = sum(g.(xs, 1.0, 2.0))', -15.25),
    ('hep = standard_module("particle-physics", "0.1")\nf(x) = sum(hep.interp_poly6_exp.([0.5, 0.25], 1.0, [2.0, 4.0], 2*x-3))\nscore = sum(f.(xs))', 22.3125),
    ('poly = standard_module("polynomials", "0.1")\nn = 2\nscore = sum(poly.legendre.(n, xs))', 11.625),
    ('dist = standard_module("distances", "0.1")\nscore = sum(dist.euclidean.([[xs[1], 0.0], [xs[2], 2.0], [xs[3], 0.0]], [[0.0, 0.0]]))', 5.5),
    ("f(x) = x + sum(xs)\nscore = sum(f.(xs))", 18.0),
    ("f(x) = 7.0\nscore = sum(f.(xs))", 21.0),
    ("f(a, x) = log(a) + x\nscore = sum(f.([2.0], xs))", 4.5 + 3 * np.log(2.0)),
    ("f(x) = xs\nscore = sum(fn(sum(_)).(f.(xs)))", 13.5),
    ("g(a, b) = a + b\nf(x) = sum(g.(xs, x))\nscore = sum(f.(xs))", 27.0),
    ("f(x) = sum([x, x + 1.0])\nscore = sum(f.(xs))", 12.0),
    ("f(x) = sum(fill(x, 2))\nscore = sum(f.(xs))", 9.0),
    ("counts = table(n = [2, 3])\nf(x) = sum(fill(x, counts.n[2]))\nscore = sum(f.(xs))", 13.5),
    ("f(x) = sum(cat(x, x + 1.0))\nscore = sum(f.(xs))", 12.0),
    ("f(x) = sum(cat([x], xs))\nscore = sum(f.(xs))", 18.0),
    ("f(x) = sum(fn(sum(_)).(cat([[x, x]], [[1.0, 2.0], [3.0, 4.0]])))\nscore = sum(f.(xs))", 39.0),
    ("f(x) = mean([x, x + 1.0])\nscore = sum(f.(xs))", 6.0),
    ("f(x) = l1norm([x, x + 1.0])\nscore = sum(f.(xs))", 12.0),
    ("f(x) = ifelse(isfinite(log10(x)), x, 0.0)\nscore = sum(f.(xs))", 4.5),
    ("f(i) = xs[i]\nscore = sum(f.([1, 2, 3]))", 4.5),
    ("f(x) = sum(get([x, x + 1.0, x + 2.0], [3, 1, 3]))\nscore = sum(f.(xs))", 25.5),
    ("f(x) = (x > 1.0) && !(x > 2.0)\nscore = sum(f.(xs))", 1),
    ("f(x) = lall([x > 1.0, x > 2.0])\nscore = sum(f.(xs))", 1),
])
def test_callable_broadcast_preserves_cells_and_captures(tmp_path, body, expected):
    _gate_engine("stablehlo")
    model = tmp_path / "model.flatppl"
    model.write_text(
        "xs = elementof(cartpow(reals, 3))\n"
        f"{body}\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        "xs = elementof(cartpow(reals, 3))\n"
        'm = load_module("model.flatppl", xs = xs)\n'
        "inputs = (xs)\noutputs = m.score\n"
    )
    src = ex.emit(query, "logdensity")
    assert ex.value(src, [[0.5, 1.5, 2.5]]) == pytest.approx(expected, rel=0, abs=1e-5)


@pytest.mark.stablehlo_only
def test_batched_posterior_preserves_composed_density_captures(tmp_path):
    _gate_engine("stablehlo")
    (tmp_path / "model.flatppl").write_text(
        "mu ~ Normal(0.0, 1.0)\n"
        "M = superpose(weighted(0.4, Normal(mu, 1.0)), weighted(0.6, Normal(3.0, 2.0)))\n"
        "y ~ iid(M, 3)\n"
        "L = likelihoodof(kernelof(record(y = y), mu = mu), record(y = [0.1, 0.2, 0.3]))\n"
        "post = bayesupdate(L, lawof(record(mu = mu)))\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        'm = load_module("model.flatppl")\n'
        "f(theta) = logdensityof(m.post, record(mu = theta))\n"
        "xs = elementof(cartpow(reals, 2))\n"
        "inputs = (xs)\noutputs = sum(f.(xs))\n"
    )
    src = ex.emit(query, "logdensity", dtype="f64")
    theta = np.array([-0.7, 1.2])
    observations = np.array([0.1, 0.2, 0.3])
    normalizer = 0.5 * np.log(2 * np.pi)
    a = np.log(0.4) - normalizer - 0.5 * (observations - theta[:, None])**2
    b = np.log(0.6) - np.log(2) - normalizer - 0.5 * ((observations - 3) / 2)**2
    logp = np.logaddexp(a, b)
    expected = (logp.sum(axis=1) - normalizer - 0.5 * theta**2).sum()
    np.testing.assert_allclose(ex.value(src, [theta]), expected, rtol=0, atol=1e-12)
    derivative = (np.exp(a - logp) * (observations - theta[:, None])).sum(axis=1) - theta
    np.testing.assert_allclose(ex.gradient(src, [theta], [0]), [derivative], rtol=0, atol=1e-12)


@pytest.mark.stablehlo_only
def test_continued_poisson_zero_variate_retains_the_rate_derivative():
    _gate_engine("stablehlo")
    directory = Path(__file__).resolve().parents[1] / "corpora/stablehlo/continued_poisson"
    src = ex.emit_concat(directory, "logdensity")
    # At x=0, §09 gives log-density = -rate, including rate=0.
    for rate in (0.0, 1.0):
        assert ex.gradient(src, [0.0, rate], [1]) == pytest.approx([-1.0])


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("invalid", [1.5, 2**31])
def test_integer_abi_accepts_integral_spelling_but_rejects_loss(invalid):
    _gate_engine("stablehlo")
    directory = Path(__file__).resolve().parents[1] / "corpora/coverage/typed_abi_inputs"
    src = ex.emit_concat(directory, "logdensity")
    assert ex.value(src, [[1.0, 2.0, 3.0], 1]) == pytest.approx(-2.112085713764618, abs=1e-6)
    with pytest.raises(ValueError, match="integer ABI"):
        ex.value(src, [[invalid, 2, 3], 1])


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("key", [(2**63 + 1, 0), (2**64 - 1, 0)])
def test_full_width_tuple_key_matches_the_same_uint64_array(key):
    _gate_engine("stablehlo")
    directory = Path(__file__).resolve().parents[1] / "corpora/stablehlo-sample/normal"
    src = ex.emit_concat(directory, "sample")
    tuple_result = ex.sample_call(src, key, [0.0, 1.0])
    array_result = ex.sample_call(src, np.asarray(key, dtype=np.uint64), [0.0, 1.0])
    for actual, expected in zip(tuple_result, array_result):
        np.testing.assert_array_equal(actual, expected)

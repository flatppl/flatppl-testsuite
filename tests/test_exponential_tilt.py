"""Analytic tilt laws, live normalizers and count-law numerical boundaries."""

import math

import numpy as np
import pytest
from scipy import special, stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def _call(tmp_path, source, precision):
    query = tmp_path / "query.flatppl"
    query.write_text(source)
    module = ex.emit(query, "logdensity", dtype=precision)
    _, _, hlo_call = ex._jax()
    return lambda *args: hlo_call(*args, source=module)


def _families():
    # Each oracle is the original log density, log MGF A(theta), and A'(theta).
    # The tests use base density + theta*T(x) - A(theta), independently of the
    # compiler's transformed-parameter formulas.
    log_mix = lambda p, t: np.logaddexp(np.log1p(-p), np.log(p) + t)
    probability = lambda p, t: special.expit(special.logit(p) + t)
    mu, cov, theta, x = (np.array([-.3, .8]), np.array([[1.4, .3], [.3, .9]]),
                          np.array([.4, -.25]), np.array([.2, -.5]))
    return [
        ("normal", "Normal(-0.7, 1.3)", "reals", .45, .8,
         stats.norm.logpdf(.8, -.7, 1.3), -.7*.45 + .5*1.3**2*.45**2, -.7 + 1.3**2*.45),
        ("mvnormal", "MvNormal([-0.3, 0.8], [[1.4, 0.3], [0.3, 0.9]])",
         "cartpow(reals, 2)", theta, x, stats.multivariate_normal.logpdf(x, mu, cov),
         mu @ theta + .5*theta @ cov @ theta, mu + cov @ theta),
        ("exponential", "Exponential(2.1)", "reals", .6, .7,
         stats.expon.logpdf(.7, scale=1/2.1), np.log(2.1/1.5), 1/1.5),
        ("gamma", "Gamma(2.7, 1.9)", "reals", .35, 1.8,
         stats.gamma.logpdf(1.8, 2.7, scale=1/1.9), 2.7*np.log(1.9/1.55), 2.7/1.55),
        ("chisquared", "ChiSquared(5.3)", "reals", -.2, 3.2,
         stats.chi2.logpdf(3.2, 5.3), -5.3/2*np.log1p(.4), 5.3/1.4),
        ("bernoulli", "Bernoulli(0.37)", "integers", .8, 1,
         stats.bernoulli.logpmf(1, .37), log_mix(.37, .8), probability(.37, .8)),
        ("binomial", "Binomial(7, 0.27)", "integers", -.65, 2,
         stats.binom.logpmf(2, 7, .27), 7*log_mix(.27, -.65), 7*probability(.27, -.65)),
        ("poisson", "Poisson(2.3)", "integers", .4, 4,
         stats.poisson.logpmf(4, 2.3), 2.3*np.expm1(.4), 2.3*np.exp(.4)),
        ("geometric", "Geometric(0.42)", "integers", -.3, 2,
         stats.geom.logpmf(3, .42), np.log(.42) - np.log1p(-.58*np.exp(-.3)),
         .58*np.exp(-.3)/(1 - .58*np.exp(-.3))),
        ("negative_binomial", "NegativeBinomial(2.7, 1.8)", "integers", .2, 3,
         stats.nbinom.logpmf(3, 2.7, 1.8/2.8), 2.7*np.log(1.8/(2.8-np.exp(.2))),
         2.7*np.exp(.2)/(2.8-np.exp(.2))),
        ("negative_binomial2", "NegativeBinomial2(4.2, 3.1)", "integers", -.3, 2,
         stats.nbinom.logpmf(2, 3.1, 3.1/7.3),
         3.1*np.log(3.1/(7.3-4.2*np.exp(-.3))),
         3.1*4.2*np.exp(-.3)/(7.3-4.2*np.exp(-.3))),
    ]


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("name,base,domain,theta,point,base_score,log_mgf,mean",
                         [pytest.param(*case, id=case[0]) for case in _families()])
def test_family_normalizers_and_slope_derivatives(
    tmp_path, precision, name, base, domain, theta, point, base_score, log_mgf, mean,
):
    _gate_engine("stablehlo")
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    integer = np.int32 if precision == "f32" else np.int64
    tolerance = 3e-5 if precision == "f32" else 1e-10
    vector = np.ndim(theta) != 0
    theta_domain = domain if vector else "reals"
    weight = "sum(theta .* _) + offset" if vector else "theta * _ + offset"
    reference = "Counting(integers)" if domain == "integers" else f"Lebesgue({domain})"
    call = _call(tmp_path,
        f"theta = external({theta_domain})\noffset = external(reals)\nx = external({domain})\n"
        f"B = {base}\nw = fn({weight})\nW = logweighted(w, B)\n"
        f"Q = normalize(W)\nT = tilt(logweighted(w, {reference}), B)\n"
        "inputs = (theta, offset, x)\n"
        "outputs = (logdensityof(Q, x), logdensityof(T, x), logdensityof(W, x))\n", precision)
    args = (jnp.asarray(theta, dtype=dtype), jnp.asarray(-1.2, dtype=dtype),
            jnp.asarray(point, dtype=integer if domain == "integers" else dtype))
    actual = jax.jit(call)(*args)
    raw = base_score + np.sum(theta * point) - 1.2
    expected = [raw - log_mgf + 1.2, raw - log_mgf + 1.2, raw]
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    gradient = jax.jit(jax.jacrev(lambda *a: jnp.stack(call(*a)), argnums=(0, 1)))(*args)
    np.testing.assert_allclose(gradient[0], [point - mean, point - mean, point],
                               rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(gradient[1], [0, 0, 1])


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_named_weight_captures_keep_live_parameters_under_vmap_and_ad(tmp_path, precision):
    _gate_engine("stablehlo")
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    tolerance = 3e-5 if precision == "f32" else 1e-10
    call = _call(tmp_path,
        "mu = external(reals)\nsigma = external(posreals)\ntheta = external(reals)\n"
        "offset = external(reals)\nx = external(reals)\nv = elementof(reals)\n"
        "linear = theta * v\nweight = functionof(linear + offset, value = v)\n"
        "alias = weight\nQ = normalize(logweighted(alias, Normal(mu, sigma)))\n"
        "inputs = (mu, sigma, theta, offset, x)\noutputs = logdensityof(Q, x)\n", precision)
    points = np.array([[-.7, 1.3, .45, -1.2, .8], [1.1, .7, -.4, 1e20, -.1],
                       [-2, 2, .25, -1e20, 1]], dtype=dtype)
    evaluate = jax.jit(jax.vmap(jax.value_and_grad(lambda *a: call(*a)[0], argnums=tuple(range(5)))))
    values, gradient = evaluate(*(jnp.asarray(column) for column in points.T))
    mu, sigma, theta, _, x = points.astype(np.float64).T
    mean = mu + theta*sigma**2
    delta = x - mean
    expected = [delta/sigma**2, -1/sigma + delta**2/sigma**3 + 2*theta*delta/sigma,
                delta, np.zeros(3), -delta/sigma**2]
    np.testing.assert_allclose(values, stats.norm.logpdf(x, mean, sigma), rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(gradient, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("coefficient", ["theta", "[theta]"])
def test_scalar_and_singleton_coefficients_broadcast_over_mvnormal(tmp_path, coefficient):
    _gate_engine("stablehlo")
    jax, _, _ = ex._jax()
    call = _call(tmp_path,
        "theta = external(reals)\nx = external(cartpow(reals, 2))\n"
        f"Q = normalize(logweighted(fn(sum({coefficient} .* _)), "
        "MvNormal([1, -1], [[2, 0.5], [0.5, 3]])))\n"
        "inputs = (theta, x)\noutputs = logdensityof(Q, x)\n", "f64")
    theta, x = .3, np.array([.2, -.5])
    cov = np.array([[2, .5], [.5, 3]])
    mean = np.array([1, -1]) + cov @ np.full(2, theta)
    value, gradient = jax.jit(jax.value_and_grad(lambda t: call(t, x)[0]))(theta)
    assert value == pytest.approx(stats.multivariate_normal.logpdf(x, mean, cov), abs=1e-11)
    assert gradient == pytest.approx(np.sum(x - mean), abs=1e-11)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("base,tail_score,original_score", [
    ("Poisson(1)", -1000., stats.poisson.logpmf(1, 1)),
    ("Geometric(0.5)", -1000.-np.log(2), stats.geom.logpmf(2, .5)),
    ("NegativeBinomial(2, 2)", -1000.+np.log(2/3), stats.nbinom.logpmf(1, 2, 2/3)),
    ("NegativeBinomial2(3, 2)", -1000.+np.log(6/5), stats.nbinom.logpmf(1, 2, 2/5)),
])
def test_count_tails_and_reversed_tilts_keep_natural_parameters(
    tmp_path, precision, base, tail_score, original_score,
):
    _gate_engine("stablehlo")
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    tolerance = 3e-5 if precision == "f32" else 1e-10
    call = _call(tmp_path,
        "theta = external(reals)\n"
        f"first = normalize(logweighted(fn(theta * _), {base}))\n"
        "restored = normalize(logweighted(fn(-theta * _), first))\n"
        "inputs = theta\noutputs = (logdensityof(first, 1), logdensityof(restored, 1))\n", precision)
    argument = jnp.asarray(-1000, dtype=dtype)
    value = jax.jit(call)(argument)
    gradient = jax.jit(jax.jacrev(lambda t: jnp.stack(call(t))))(argument)
    np.testing.assert_allclose(value, [tail_score, original_score], rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(gradient, [1, 0], rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("base,parameter,point,derivative", [
    ("Bernoulli(a)", 0., 0, -np.exp(.5)),
    ("Bernoulli(a)", 1., 1, np.exp(-.5)),
    ("Binomial(7, a)", 0., 0, -7*np.exp(.5)),
    ("Binomial(7, a)", 1., 7, 7*np.exp(-.5)),
    ("Geometric(a)", 1., 0, np.exp(.5)),
    ("Poisson(a)", 0., 0, -np.exp(.5)),
])
def test_supported_count_endpoints_retain_one_sided_derivatives(
    tmp_path, precision, base, parameter, point, derivative,
):
    _gate_engine("stablehlo")
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    tolerance = 3e-5 if precision == "f32" else 1e-10
    call = _call(tmp_path,
        "a = external(nonnegreals)\ntheta = external(reals)\n"
        f"Q = normalize(logweighted(fn(theta * _), {base}))\n"
        f"inputs = (a, theta)\noutputs = logdensityof(Q, {point})\n", precision)
    args = (jnp.asarray(parameter, dtype=dtype), jnp.asarray(.5, dtype=dtype))
    value, gradient = jax.jit(jax.value_and_grad(lambda *a: call(*a)[0], argnums=(0, 1)))(*args)
    assert value == 0
    np.testing.assert_allclose(gradient, [derivative, 0], rtol=tolerance, atol=tolerance)
    if base.startswith(("Bernoulli", "Binomial")):
        for theta in (-1000, 1000):
            assert jax.jit(call)(args[0], jnp.asarray(theta, dtype=dtype))[0] == 0


@pytest.mark.stablehlo_only
def test_negative_binomial_large_shape_retains_small_count_mass(tmp_path):
    _gate_engine("stablehlo")
    jax, _, _ = ex._jax()
    call = _call(tmp_path,
        "a = external(posreals)\ntheta = external(reals)\nx = external(integers)\n"
        "Q = normalize(logweighted(fn(theta * _), NegativeBinomial(a, 2)))\n"
        "inputs = (a, theta, x)\noutputs = logdensityof(Q, x)\n", "f64")
    theta = np.log(3.) - 710
    rate = np.exp(np.log(1e308) - 710)
    for count in (0, 1, 2):
        value = jax.jit(call)(1e308, theta, np.int64(count))[0]
        # For k<=2 the exact rising product differs from a**k by <=1e-308.
        # The next term in a*log(1-exp(-710)) is also below 1e-308.
        expected = -rate + count*(np.log(1e308)-710) - math.lgamma(count+1)
        assert value == pytest.approx(expected, abs=2e-10)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,count", [("f32", 10**8), ("f64", 10**18)])
def test_negative_binomial_large_count_uses_the_finite_product_oracle(tmp_path, precision, count):
    _gate_engine("stablehlo")
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    integer = np.int32 if precision == "f32" else np.int64
    tolerance = 3e-5 if precision == "f32" else 1e-10
    call = _call(tmp_path,
        "beta = external(posreals)\nx = external(integers)\n"
        "Q = normalize(logweighted(fn(0 * _), NegativeBinomial(2, beta)))\n"
        "inputs = (beta, x)\noutputs = logdensityof(Q, x)\n", precision)
    beta = float(dtype(1/count))
    value = jax.jit(call)(jnp.asarray(beta, dtype=dtype), jnp.asarray(count, dtype=integer))[0]
    # At shape two the binomial coefficient is exactly k+1.
    expected = np.log(count+1) + 2*np.log(beta) - (count+2)*np.log1p(beta)
    assert value == pytest.approx(expected, rel=tolerance, abs=tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,mu,shape", [("f32", 1e38, 1e-10), ("f64", 1e300, 1e-100)])
def test_nb2_underflowing_shape_mean_ratio_retains_density_and_theta_gradient(
    tmp_path, precision, mu, shape,
):
    _gate_engine("stablehlo")
    jax, jnp, _ = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    integer = np.int32 if precision == "f32" else np.int64
    tolerance = 4e-5 if precision == "f32" else 2e-10
    call = _call(tmp_path,
        "mu = external(posreals)\nshape = external(posreals)\ntheta = external(reals)\n"
        "x = external(integers)\n"
        "Q = normalize(logweighted(fn(theta * _), NegativeBinomial2(mu, shape)))\n"
        "inputs = (mu, shape, theta, x)\noutputs = logdensityof(Q, x)\n", precision)
    evaluate = jax.jit(jax.value_and_grad(lambda m, s, t, x: call(m, s, t, x)[0], argnums=2))
    mu, shape = float(dtype(mu)), float(dtype(shape))
    for theta in (0., -.3):
        theta = float(dtype(theta))
        for count in (0, 1):
            args = tuple(jnp.asarray(v, dtype=dtype) for v in (mu, shape, theta))
            value, gradient = evaluate(*args, jnp.asarray(count, dtype=integer))
            # psi/mu is below the source precision. Its omitted contribution
            # is negligible here; finite k=0/1 products avoid gamma differences.
            log_success = np.log(shape)-np.log(mu) if theta == 0 else np.log(-np.expm1(theta))
            expected = shape*log_success + (np.log(shape) + theta if count else 0)
            mean = mu if theta == 0 else shape/np.expm1(-theta)
            assert value == pytest.approx(expected, rel=tolerance, abs=0)
            assert gradient == pytest.approx(count-mean, rel=tolerance, abs=0)

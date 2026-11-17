"""Execute HEP samplers against independently checked distribution CDFs."""

import numpy as np
import pytest
from scipy import special, stats

from flatppl_testsuite.sampler_sweep.checks import build_cdf
from flatppl_testsuite.sampler_sweep.oracle import FAMILIES
from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

N = 8192
SLUGS = ["crystalball", "doublecrystalball", "argus", "relbreitwigner",
         "voigtian", "landau", "bifurcatednormal"]


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("slug", SLUGS)
def test_hep_sampling_distribution_and_key_continuation(tmp_path, precision, slug):
    _gate_engine("stablehlo")
    family = next(f for f in FAMILIES if f.slug == slug)
    assert family.scipy is not None
    query = tmp_path / (slug + ".flatppl")
    query.write_text(
        'h = standard_module("particle-physics", "0.1")\n'
        f"outputs = rand(iid(h.{family.measure}, {N}), rnginit(271828))\n"
    )
    source = ex.emit(query, "sample", dtype=precision, restrict_enzyme_compatible=False)
    key = ex.rng_state(271828)
    draws, successor = ex.sample_call(source, key)
    repeated, repeated_key = ex.sample_call(source, key)
    np.testing.assert_array_equal(draws, repeated)
    np.testing.assert_array_equal(successor, repeated_key)
    assert not np.array_equal(key, successor)
    later, later_key = ex.sample_call(source, successor)
    assert not np.array_equal(later, draws)
    assert not np.array_equal(later_key, successor)
    assert draws.shape == (N,)
    assert np.isfinite(draws).all()
    cdf = build_cdf(("dist", *family.scipy))
    u = np.asarray(cdf(draws.astype(float)))
    assert np.isfinite(u).all()
    assert np.all((u >= 0) & (u <= 1))
    ordered = np.sort(u)
    ks = max(np.max(np.arange(1, N + 1) / N - ordered),
             np.max(ordered - np.arange(N) / N))
    # DKW with a union bound across all seven laws and both precisions.
    threshold = np.sqrt(np.log(2 * 14 / 1e-6) / (2 * N))
    assert ks < threshold
    assert abs(u.mean() - .5) < 5 / np.sqrt(12 * N)
    assert abs(np.corrcoef(u[:-1], u[1:])[0, 1]) < 5 / np.sqrt(N - 1)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("name,arguments,bad", [
    ("CrystalBall", "0., 1., 1.5, p", 1.),
    ("DoubleSidedCrystalBall", "0., 1., 1., 1.5, 1.5, 3., p", 1.),
    ("Argus", "1., 0., p", 0.),
    ("RelativisticBreitWigner", "1., p", 0.),
    ("Voigtian", "0., 1., p", 0.),
    ("Landau", "0., p", 0.),
    ("BifurcatedNormal", "0., 1., p", 0.),
])
def test_hep_sampler_invalid_parameters_report_status(tmp_path, name, arguments, bad):
    _gate_engine("stablehlo")
    query = tmp_path / "invalid.flatppl"
    query.write_text(
        'h = standard_module("particle-physics", "0.1")\n'
        f"p = external(reals)\ninputs = p\noutputs = rand(h.{name}({arguments}), rnginit(1))\n"
    )
    source = ex.emit(query, "sample", dtype="f64", restrict_enzyme_compatible=False)
    jax, jnp, _ = ex._jax()
    args = ex._arguments(jnp, source, [ex.rng_state(1), bad])
    _, status = jax.jit(ex.call_with_status(source))(*args)
    assert int(status) == 9


@pytest.mark.stablehlo_only
def test_breit_wigner_sampler_retains_implicit_quantile_derivatives(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "rbw-gradient.flatppl"
    query.write_text(
        'h = standard_module("particle-physics", "0.1")\n'
        "m = external(posreals)\nw = external(posreals)\ninputs = (m, w)\n"
        "outputs = rand(h.RelativisticBreitWigner(m, w), rnginit(1))\n"
    )
    source = ex.emit(query, "sample", dtype="f64", restrict_enzyme_compatible=False)
    for mean, width in [(3., .4), (.01, 10.)]:
        key = ex.rng_state(123)
        x, _ = ex.sample_call(source, key, [mean, width])
        density = stats.rel_breitwigner.pdf(x, mean / width, scale=width)
        expected = []
        for index, value in enumerate([mean, width]):
            step = value * 1e-5
            left, right = [mean, width], [mean, width]
            left[index] -= step
            right[index] += step
            low = stats.rel_breitwigner.cdf(x, left[0] / left[1], scale=left[1])
            high = stats.rel_breitwigner.cdf(x, right[0] / right[1], scale=right[1])
            expected.append(-(high - low) / (2 * step * density))
        np.testing.assert_allclose(ex.gradient(source, [key, mean, width], [1, 2]), expected,
                                   rtol=3e-7, atol=1e-8)


@pytest.mark.stablehlo_only
def test_unselected_crystal_ball_tails_keep_finite_sample_derivatives(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "dscb-gradient.flatppl"
    names = ["m", "sl", "sr", "al", "ar", "nl", "nr"]
    query.write_text(
        'h = standard_module("particle-physics", "0.1")\n'
        + "\n".join(f"{name} = external(reals)" for name in names)
        + "\ninputs = (" + ", ".join(names) + ")\n"
        + "outputs = rand(h.DoubleSidedCrystalBall(" + ", ".join(names) + "), rnginit(1))\n"
    )
    source = ex.emit(query, "sample", dtype="f64", restrict_enzyme_compatible=False)
    point = [.3, .7, 1.1, 5., 5., 1.01, 1.01]
    key = ex.rng_state(123)
    actual = ex.gradient(source, [key, *point], list(range(1, 8)))
    expected = []
    for index, value in enumerate(point):
        step = 1e-5 * max(1, abs(value))
        left, right = point.copy(), point.copy()
        left[index] -= step
        right[index] += step
        low, _ = ex.sample_call(source, key, left)
        high, _ = ex.sample_call(source, key, right)
        expected.append(float((high - low) / (2 * step)))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)


@pytest.mark.stablehlo_only
def test_argus_sampler_preserves_varying_parameter_laws(tmp_path):
    _gate_engine("stablehlo")
    query = tmp_path / "argus-rows.flatppl"
    query.write_text(
        'h = standard_module("particle-physics", "0.1")\n'
        "m = external(cartpow(reals, 2))\nc = external(cartpow(reals, 2))\n"
        "p = external(cartpow(reals, 2))\ninputs = (m, c, p)\n"
        "outputs = rand(h.Argus.(m, c, p), rnginit(1))\n"
    )
    source = ex.emit(query, "sample", dtype="f64", restrict_enzyme_compatible=False)
    jax, jnp, _ = ex._jax()
    parameters = [np.array([1.5, 2.5]), np.array([-20., -2.]), np.array([.5, 1.3])]
    args = ex._arguments(jnp, source, [ex.rng_state(123), *parameters])

    def chain(key, *parameters):
        def step(key, _):
            draws, successor = ex._outputs(source, key, *parameters)
            return successor, draws

        return jax.lax.scan(step, key, None, length=N)

    _, draws = ex._jit_checked(source, chain)(*args)
    draws = np.asarray(draws)
    assert draws.shape == (N, 2)
    for col in range(2):
        m, c, p = [row[col] for row in parameters]
        t = (1 - draws[:, col] / m) * (1 + draws[:, col] / m)
        u = np.sort(1 - special.gammainc(p + 1, -c * t) / special.gammainc(p + 1, -c))
        ks = max(np.max(np.arange(1, N + 1) / N - u), np.max(u - np.arange(N) / N))
        assert ks < np.sqrt(np.log(4 / 1e-6) / (2 * N))

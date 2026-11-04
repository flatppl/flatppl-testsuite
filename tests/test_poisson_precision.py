"""Poisson log-density must retain precision near large effective counts."""

import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("distribution, domain", [
    ("Poisson", "integers"), ("hep.ContinuedPoisson", "reals"),
])
def test_poisson_large_count_values_and_rate_gradients(tmp_path, distribution, domain):
    _gate_engine("stablehlo")
    (tmp_path / "model.flatppl").write_text(
        'hep = standard_module("particle-physics", "0.1")\n'
        "rate = elementof(nonnegreals)\n"
        f"M = {distribution}(rate)\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        f"x = elementof({domain})\nrate = elementof(nonnegreals)\n"
        'm = load_module("model.flatppl", rate = rate)\n'
        "inputs = (x, rate)\noutputs = logdensityof(m.M, x)\n"
    )
    src = ex.emit(query, "logdensity", dtype="f64")
    # 80-digit evaluation of x*log(rate)-rate-loggamma(x+1), not pyhf's
    # cancellation-prone binary64 evaluation. Keep both inputs at runtime.
    points = [(16, 16, -2.310440550244173),
              (16, 24, -3.822998820513543),
              (16, 64, -28.129730772325924),
              (482465, 285686.965488165, -56046.72978376564),
              (10**12, 10**12, -14.73444909116903),
              (10**12, 10**12 + 10**6, -15.234448757835947)]
    if domain == "reals":
        points.append((189281768783.48367, 189281768783.48367, -13.902189824170838))
    assert ex.value(src, [0, 0]) == 0
    assert ex.gradient(src, [0, 0], [1]) == pytest.approx([-1.0])
    for x, rate, expected in points:
        assert ex.value(src, [x, rate]) == pytest.approx(expected, rel=2e-15, abs=1e-12)
        assert ex.gradient(src, [x, rate], [1]) == pytest.approx([x / rate - 1], abs=1e-14)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("dtype, rate, expected", [
    ("f64", 1e-300, -45931251.09257462),
    ("f32", 1.0000000031710769e-30, -5188274.293638312),
])
def test_poisson_rate_gradient_stays_finite_at_tiny_rates(tmp_path, dtype, rate, expected):
    # k/rate - 1 is finite here, but a reverse pass that forms k**2/rate overflows.
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "x = elementof(integers)\nrate = elementof(nonnegreals)\n"
        "inputs = (x, rate)\noutputs = logdensityof(Poisson(rate), x)\n"
    )
    src = ex.emit(query, "logdensity", dtype=dtype)
    k = 65535
    # Expected value: 60-digit k*log(rate) - rate - log(k!) at the dtype's rate.
    assert ex.value(src, [k, rate]) == pytest.approx(expected, rel=1e-6 if dtype == "f32" else 1e-14)
    assert ex.gradient(src, [k, rate], [1]) == pytest.approx([k / rate - 1], rel=1e-6 if dtype == "f32" else 1e-14)

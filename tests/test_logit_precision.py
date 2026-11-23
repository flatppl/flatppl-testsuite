"""Bernoulli and Binomial logits retain both tail masses and their derivatives."""

import math

import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("law, trials", [("Bernoulli(p)", 1), ("Binomial(5, p)", 5)])
@pytest.mark.parametrize("dtype, magnitude, rtol", [("f32", 70, 2e-5), ("f64", 700, 2e-13)])
def test_logit_likelihood_tails(tmp_path, law, trials, dtype, magnitude, rtol):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "z = elementof(reals)\nk = elementof(integers)\ninputs = (z, k)\n"
        f"p = invlogit(z)\noutputs = logdensityof({law}, k)\n"
    )
    src = ex.emit(query, "logdensity", dtype=dtype)
    for z in (-magnitude, -40, 0, 40, magnitude):
        t = math.exp(-abs(z))
        for count, sign in [(0, -1), (trials, 1)]:
            # At either endpoint the binomial coefficient is one. These are
            # trials copies of log(sigmoid(sign*z)) and its derivative in z.
            signed = sign * z
            value = trials * (min(signed, 0) - math.log1p(t))
            gradient = trials * sign * (t if signed >= 0 else 1) / (1 + t)
            assert ex.value(src, [z, count]) == pytest.approx(value, rel=rtol, abs=0)
            assert ex.gradient(src, [z, count], [0]) == pytest.approx([gradient], rel=rtol, abs=0)

"""Integer-domain count densities are zero outside their support (§08): Geometric,
NegativeBinomial and NegativeBinomial2 on `nonnegintegers`, Binomial on
`interval(0, n)`, Poisson on `nonnegintegers`. An integer point below zero (or
above `n` for Binomial) scores -inf, never a finite number or NaN.

In-support values are the closed-form log-pmfs below. The JS engine gives the same
numbers and -inf at every out-of-support point (Geometric(0.3) at 2:
-1.9173226922034008, NegativeBinomial(2, 0.5) at 2: -1.9095425048844383,
NegativeBinomial2(3, 2) at 2: -1.7556204226121819)."""

from math import inf, lgamma, log

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


def log_choose(n, k):
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def geometric(p, k):
    # Failures before the first success: p (1 - p)^k.
    return log(p) + k * log(1 - p)


def negative_binomial(alpha, beta, k):
    # Gamma-Poisson with shape alpha and rate beta.
    return (lgamma(k + alpha) - lgamma(alpha) - lgamma(k + 1)
            + alpha * log(beta / (beta + 1)) - k * log(beta + 1))


def negative_binomial2(mu, psi, k):
    return (lgamma(k + psi) - lgamma(psi) - lgamma(k + 1)
            + k * log(mu / (mu + psi)) + psi * log(psi / (mu + psi)))


def binomial(n, p, k):
    return log_choose(n, k) + k * log(p) + (n - k) * log(1 - p)


def poisson(rate, k):
    return k * log(rate) - rate - lgamma(k + 1)


# name: (distribution, in-support oracle)
DISTRIBUTIONS = {
    "geometric": ("Geometric(0.3)", lambda k: geometric(0.3, k)),
    "negative_binomial": ("NegativeBinomial(2.0, 0.5)", lambda k: negative_binomial(2.0, 0.5, k)),
    "negative_binomial2": ("NegativeBinomial2(3.0, 2.0)", lambda k: negative_binomial2(3.0, 2.0, k)),
    "binomial": ("Binomial(5, 0.3)", lambda k: binomial(5, 0.3, k)),
    "poisson": ("Poisson(3.0)", lambda k: poisson(3.0, k)),
}

# Out-of-support integer points per distribution; every one scores -inf.
OUTSIDE = {"geometric": [-2, -1], "negative_binomial": [-2, -1],
           "negative_binomial2": [-2, -1], "binomial": [-1, 6], "poisson": [-2, -1]}
INSIDE = [0, 2]


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("name", sorted(DISTRIBUTIONS))
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_count_density_is_zero_outside_its_support(tmp_path, name, precision):
    _gate_engine("stablehlo")
    distribution, oracle = DISTRIBUTIONS[name]
    query = tmp_path / "query.flatppl"
    query.write_text(f"k = elementof(integers)\ninputs = (k)\n"
                     f"outputs = logdensityof({distribution}, k)\n")
    source = ex.emit(query, "logdensity", dtype=precision)
    for k in OUTSIDE[name]:
        assert ex.value(source, [k]) == -inf, f"{distribution} at {k}"
    rtol = 1e-6 if precision == "f32" else 1e-13
    for k in INSIDE:
        np.testing.assert_allclose(ex.value(source, [k]), oracle(k), rtol=rtol)

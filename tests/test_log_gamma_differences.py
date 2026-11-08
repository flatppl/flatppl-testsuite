"""Densities whose normalizers are differences of log-gammas keep their precision
at large parameters (§08): Beta, Binomial, NegativeBinomial, NegativeBinomial2 and
Dirichlet. Each log-gamma there is far larger than the difference, so subtracting
rounded log-gammas loses digits that the density never had to lose.

The tolerance is 4 eps times the summed magnitude of the density's own terms in
a stable form (log-beta, the count terms and the log-probability terms), so it
admits the rounding of those terms and nothing from the cancelled log-gammas.
Values and slopes come from mpmath at 50 digits."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

MODELS = {
    "beta": ("a = elementof(posreals)\nb = elementof(posreals)\nx = elementof(interval(0.0, 1.0))\n"
             "inputs = (a, b, x)\noutputs = logdensityof(Beta(a, b), x)\n"),
    "binomial": ("n = elementof(nonnegintegers)\nk = elementof(nonnegintegers)\n"
                 "p = elementof(interval(0.0, 1.0))\n"
                 "inputs = (n, k, p)\noutputs = logdensityof(Binomial(n, p), k)\n"),
    "negative_binomial": ("a = elementof(posreals)\nb = elementof(posreals)\nk = elementof(nonnegintegers)\n"
                          "inputs = (a, b, k)\noutputs = logdensityof(NegativeBinomial(a, b), k)\n"),
    "negative_binomial2": ("mu = elementof(posreals)\npsi = elementof(posreals)\nk = elementof(nonnegintegers)\n"
                           "inputs = (mu, psi, k)\noutputs = logdensityof(NegativeBinomial2(mu, psi), k)\n"),
    "dirichlet": ("a = elementof(cartpow(posreals, 4))\nx = elementof(cartpow(reals, 4))\n"
                  "inputs = (a, x)\noutputs = logdensityof(Dirichlet(a), x)\n"),
}

X4 = [0.1875, 0.296875, 0.1171875, 0.3984375]

# name: (model, point, log density, summed term magnitude, shape-parameter argnum, its slope)
CASES = {
    "beta_both_large": ("beta", [25000.0, 75000.0, 0.25], 5.674508804955166, 112469.35547969949,
                        [0, 1], [1.5000124999999979e-05, 1.666673148148148e-06]),
    "beta_one_large": ("beta", [3.0, 40000.0, 0.0001], 8.676051264879456, 53.517612769452185,
                       [0, 1], [0.4635725240944183, -2.5006875255236856e-05]),
    "beta_mid": ("beta", [900.0, 2700.0, 0.25], 4.012293937165181, 4049.477382325442,
                 [0, 1], [0.0004167631172712989, 4.6301297439307535e-05]),
    "binomial": ("binomial", [1000000, 300000, 0.3], -7.046370251546539, 1221749.1887626513, [], []),
    "binomial_mid": ("binomial", [20000, 6000, 0.3], -5.090374109943563, 24449.288783188367, [], []),
    "negative_binomial": ("negative_binomial", [30000.0, 1.5, 21000], -20.819852082040054,
                          69134.60747441533, [0], [0.019809490101831386]),
    "negative_binomial_mid": ("negative_binomial", [3000.0, 1.5, 2100], -6.475095636329982,
                              6923.97371246849, [0], [0.019871260802523338]),
    "negative_binomial2": ("negative_binomial2", [20000.0, 30000.0, 21000], -20.819852082040054,
                           69134.60747441533, [1], [-0.00019050989816861454]),
    "negative_binomial2_mid": ("negative_binomial2", [2000.0, 3000.0, 2100], -6.475095636329982,
                               6923.97371246849, [1], [-0.00012873919747666358]),
    "dirichlet": ("dirichlet", [[30000.0, 50000.0, 20000.0, 70000.0], X4], -77.80525612060192,
                  436381.13684707636, [0],
                  [[0.060638347396340005, 0.009338386282863547, -0.0038918402921570903, -0.03289723449959305]]),
    "dirichlet_mid": ("dirichlet", [[300.0, 500.0, 200.0, 700.0], X4], 10.398825924907195,
                      4363.202088597957, [0],
                      [[0.06199806792587737, 0.010037514279963047, -0.00170596247513253, -0.03248112689423167]]),
    # Concentrations below one, where Stirling's series diverges and lgamma is used.
    "dirichlet_small": ("dirichlet", [[0.1, 0.5, 2.0, 0.25], X4], -2.8938640403962257,
                        8.501772671811404, [0],
                        [[9.611519036925426, 1.6108064519142122, -1.705023867829854, 4.1689894322669945]]),
    "dirichlet_mixed": ("dirichlet", [[0.1, 5000.0, 0.5, 30000.0], X4], -19312.85975253342,
                        48042.39186831127, [0],
                        [[19.212884704483745, 0.7315689053682042, 10.282636160848355, -0.7660344274359849]]),
    "beta_small": ("beta", [0.1, 0.5, 0.25], -1.035337775355879, 3.818349697823463,
                   [0, 1], [7.496841365397995, 0.13520873967645217]),
    "beta_mixed": ("beta", [0.1, 50000.0, 0.0001], 2.118420599815403, 14.460192069741726,
                   [0, 1], [12.033184852829843, -9.800498233326235e-05]),
}

EPS = {"f32": 2.0 ** -23, "f64": 2.0 ** -52}


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_log_gamma_difference_keeps_its_precision(tmp_path, case, precision):
    _gate_engine("stablehlo")
    model, point, value, magnitude, argnums, slopes = CASES[case]
    query = tmp_path / "query.flatppl"
    query.write_text(MODELS[model])
    source = ex.emit(query, "logdensity", dtype=precision)
    eps = EPS[precision]
    np.testing.assert_allclose(ex.value(source, point), value, rtol=0, atol=4 * eps * magnitude)
    # A shape slope is a digamma difference plus logarithms of order ten that
    # cancel to a small number, so its rounding scales with eps, not with the slope.
    if argnums:
        for actual, expected in zip(ex.gradient(source, point, argnums), slopes, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5 if precision == "f32" else 1e-12,
                                       atol=64 * eps)

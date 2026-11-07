"""Every §08 multivariate density scores under broadcast: a dotted constructor with
one parameter cell per observation, `iid`, and `ksuperpose`. Each density and its
slope in `z` is checked against scipy, or against the n = 2 LKJ closed form, in
float32 and float64."""

import numpy as np
import pytest
from scipy.special import betaln, logsumexp
from scipy.stats import dirichlet, invwishart, multinomial, wishart

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

Z = 1.3
SCALE = np.array([[2.0, 0.3], [0.3, 1.0]])
SPD = [np.array([[1.0, 0.2], [0.2, 2.0]]), np.array([[3.0, -0.5], [-0.5, 1.5]])]
R = [0.3, -0.2]
SIMPLEX = [[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]

_S = "rowstack([[2.0, 0.3], [0.3, 1.0]])"
_SPD = "[rowstack([[1.0, 0.2], [0.2, 2.0]]), rowstack([[3.0, -0.5], [-0.5, 1.5]])]"
_CORR = "[rowstack([[1.0, 0.3], [0.3, 1.0]]), rowstack([[1.0, -0.2], [-0.2, 1.0]])]"
_CHOL = ("[rowstack([[1.0, 0.0], [0.3, 0.9539392014169457]]), "
         "rowstack([[1.0, 0.0], [-0.2, 0.9797958971132712]])]")


def lkj(r, eta):
    """log LKJ(C | eta) for n = 2, C = [[1, r], [r, 1]]."""
    return (eta - 1) * np.log(1 - r * r) - (2 * eta - 1) * np.log(2) - betaln(eta, eta)


def lkj_cholesky(r, eta):
    """log LKJCholesky(L | eta) for n = 2: (2 eta - 2) log L22 minus the LKJ normalizer."""
    return (2 * eta - 2) * np.log(np.sqrt(1 - r * r)) - (2 * eta - 1) * np.log(2) - betaln(eta, eta)


CASES = {
    "dirichlet_cells": (
        f"logdensityof(Dirichlet.([[1.0, 2.0, 3.0], [0.5, 0.5, 2.0]] .* z), {SIMPLEX})",
        lambda z: sum(dirichlet.logpdf(x, np.array(a) * z)
                      for x, a in zip(SIMPLEX, [[1.0, 2.0, 3.0], [0.5, 0.5, 2.0]])),
    ),
    "multinomial_cells": (
        f"logdensityof(Multinomial.(5, {SIMPLEX}), [[1, 2, 2], [0, 1, 4]]) + 0.0 * z",
        lambda z: multinomial.logpmf([1, 2, 2], 5, SIMPLEX[0]) + multinomial.logpmf([0, 1, 4], 5, SIMPLEX[1]),
    ),
    "lkj_iid": (f"logdensityof(iid(LKJ(2, z), 2), {_CORR})", lambda z: sum(lkj(r, z) for r in R)),
    "lkj_cells": (f"logdensityof(LKJ.(2, [z, 2.0 * z]), {_CORR})",
                  lambda z: lkj(R[0], z) + lkj(R[1], 2 * z)),
    "lkj_cholesky_iid": (f"logdensityof(iid(LKJCholesky(2, z), 2), {_CHOL})",
                         lambda z: sum(lkj_cholesky(r, z) for r in R)),
    "lkj_cholesky_cells": (f"logdensityof(LKJCholesky.(2, [z, 2.0 * z]), {_CHOL})",
                           lambda z: lkj_cholesky(R[0], z) + lkj_cholesky(R[1], 2 * z)),
    "wishart_iid": (f"logdensityof(iid(Wishart(z + 3.0, {_S}), 2), {_SPD})",
                    lambda z: sum(wishart.logpdf(x, df=z + 3.0, scale=SCALE) for x in SPD)),
    "wishart_cells": (f"logdensityof(Wishart.([z + 3.0, z + 4.0], fill({_S}, 2)), {_SPD})",
                      lambda z: wishart.logpdf(SPD[0], df=z + 3.0, scale=SCALE)
                      + wishart.logpdf(SPD[1], df=z + 4.0, scale=SCALE)),
    "inverse_wishart_iid": (f"logdensityof(iid(InverseWishart(z + 3.0, {_S}), 2), {_SPD})",
                            lambda z: sum(invwishart.logpdf(x, df=z + 3.0, scale=SCALE) for x in SPD)),
    "inverse_wishart_cells": (
        f"logdensityof(InverseWishart.([z + 3.0, z + 4.0], fill({_S}, 2)), {_SPD})",
        lambda z: invwishart.logpdf(SPD[0], df=z + 3.0, scale=SCALE)
        + invwishart.logpdf(SPD[1], df=z + 4.0, scale=SCALE),
    ),
    "lkj_mixture": (
        "logdensityof(ksuperpose(LKJ, [0.3, 0.7], n = [2, 2], eta = [z, 2.0]), "
        "rowstack([[1.0, 0.3], [0.3, 1.0]]))",
        lambda z: logsumexp(np.log([0.3, 0.7]) + [lkj(0.3, z), lkj(0.3, 2.0)]),
    ),
    "wishart_mixture": (
        f"logdensityof(ksuperpose(Wishart, [0.3, 0.7], nu = [z + 3.0, 5.0], scale = fill({_S}, 2)), "
        "rowstack([[1.0, 0.2], [0.2, 2.0]]))",
        lambda z: logsumexp(np.log([0.3, 0.7]) + [wishart.logpdf(SPD[0], df=z + 3.0, scale=SCALE),
                                                  wishart.logpdf(SPD[0], df=5.0, scale=SCALE)]),
    ),
}


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,rtol", [("f32", 3e-6), ("f64", 1e-12)])
@pytest.mark.parametrize("case", sorted(CASES))
def test_multivariate_density_scores_each_cell(tmp_path, case, precision, rtol):
    _gate_engine("stablehlo")
    expr, oracle = CASES[case]
    query = tmp_path / "query.flatppl"
    query.write_text(f"z = elementof(posreals)\nlp = {expr}\ninputs = (z)\noutputs = (lp)\n")
    source = ex.emit(query, "logdensity", dtype=precision)
    np.testing.assert_allclose(ex.value(source, [Z]), oracle(Z), rtol=rtol, atol=rtol)
    h = 1e-6
    slope = (oracle(Z + h) - oracle(Z - h)) / (2 * h)
    np.testing.assert_allclose(ex.gradient(source, [Z], [0])[0], slope,
                               rtol=max(rtol, 1e-7) * 100, atol=1e-6 if precision == "f32" else 1e-8)

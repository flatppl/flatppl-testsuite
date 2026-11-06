"""A NegativeBinomial2 density broadcasts over a vector of counts (§08)."""

import numpy as np
import pytest
from scipy import special, stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

MU = np.array([2.0, 5.0, 0.7])
COUNTS = np.array([1, 7, 0])


def _oracle(psi):
    """log NB2 and d/dpsi, with NB2(mu, psi) = NegBinomial(n = psi, p = psi / (psi + mu))."""
    value = stats.nbinom.logpmf(COUNTS, psi, psi / (psi + MU)).sum()
    slope = (special.digamma(COUNTS + psi) - special.digamma(psi)
             + np.log(psi / (psi + MU)) + (MU - COUNTS) / (psi + MU)).sum()
    return value, slope


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-5), ("f64", 1e-12)])
def test_negative_binomial2_broadcast_scores_each_count(tmp_path, precision, tolerance):
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "psi = elementof(posreals)\ninputs = psi\noutputs = logdensityof("
        "NegativeBinomial2.([2.0, 5.0, 0.7], psi), [1, 7, 0])\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    for psi in [0.4, 3.0, 25.0]:
        value, slope = _oracle(psi)
        np.testing.assert_allclose(ex.value(source, [psi]), value, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(ex.gradient(source, [psi], [0])[0], slope,
                                   rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 3e-6), ("f64", 1e-13)])
def test_negative_binomial2_family_superposition(tmp_path, precision, tolerance):
    """ksuperpose scores every NegativeBinomial2 member through the same broadcast."""
    _gate_engine("stablehlo")
    query = tmp_path / "query.flatppl"
    query.write_text(
        "psi = elementof(posreals)\ninputs = psi\noutputs = logdensityof("
        "ksuperpose(NegativeBinomial2, [0.3, 1.2], mu = [1.0, 2.0], psi = psi), 3)\n"
    )
    source = ex.emit(query, "logdensity", dtype=precision)
    weights, means = np.array([0.3, 1.2]), np.array([1.0, 2.0])
    for psi in [0.5, 2.0, 9.0]:
        expected = special.logsumexp(np.log(weights) + stats.nbinom.logpmf(3, psi, psi / (psi + means)))
        np.testing.assert_allclose(ex.value(source, [psi]), expected, rtol=tolerance, atol=tolerance)

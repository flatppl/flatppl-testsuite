"""Independent oracle for cov_b_mass_peak (scipy).

Mixture log-likelihood plus the Beta prior:

    lp(f) = Beta(f | 2, 2).logpdf
          + sum_i log(f * pdf_CB(m_i) + (1 - f) * pdf_Argus(m_i))

Parameter mapping to §09 (verified against a direct transcription of
the §09 formulas, agreement ~4e-9 relative from quad tolerance):

* `CrystalBall(m0, sigma, alpha, n)` = scipy `crystalball(beta = alpha,
  m = n, loc = m0, scale = sigma)` (left power-law tail).
* `Argus(resonance, slope, power = 0.5)` = scipy `argus(chi, scale =
  resonance)` with slope = -chi^2 / 2, so chi = sqrt(40) here.

STATUS: the rust determiniser REFUSES any standard-module member on
this path ("cross-module ref could not be resolved against the module
bundle", verified live 2026-09-01) — the same class the examples corpus
records for `load_module`. flatppl-js registers all eight §09
particle-physics distributions density-only, so the pin is
determiniser-side. `allow_skip: true`; the frozen values are real
oracle values and take over when module bundles lower.
"""
import numpy as np
from scipy import stats
from scipy.stats import argus, crystalball

M_OBS = np.array([
    4.70412, 4.71995, 4.80547, 4.84687, 4.91867, 5.01576, 5.04388,
    5.06559, 5.06809, 5.14104, 5.1471, 5.15618, 5.15856, 5.20357,
    5.21367, 5.22326, 5.26491, 5.26499, 5.27234, 5.27584, 5.27706,
    5.27769, 5.27792, 5.27928, 5.27954, 5.28146, 5.28253, 5.28356,
    5.28574, 5.28812,
])
_CHI = np.sqrt(40.0)


def oracle(point: dict) -> float:
    f = float(point["f"])
    ps = crystalball.pdf(M_OBS, 1.5, 3.0, loc=5.279, scale=0.003)
    pb = argus.pdf(M_OBS, _CHI, scale=5.29)
    lik = np.log(f * ps + (1.0 - f) * pb).sum()
    return float(lik + stats.beta.logpdf(f, 2.0, 2.0))

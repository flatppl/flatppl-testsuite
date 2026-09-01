"""Independent oracle for cov_b_mass_peak (scipy).

Mixture log-likelihood plus the Beta prior:

    lp(f) = Beta(f | 2, 2).logpdf
          + sum_i log(f * pdf_CB(m_i) + (1 - f) * pdf_Argus(m_i))

Parameter mapping to §09, re-verified 2026-09-01 against a direct
transcription of the §09 formulas with CLOSED-FORM normalizers (not
`quad`): all four points agree to 0–4e-16 relative. The earlier ~4e-9
residual was `quad`'s own tolerance on the Crystal Ball normalizer.

* `CrystalBall(m0, sigma, alpha, n)` = scipy `crystalball(beta = alpha,
  m = n, loc = m0, scale = sigma)` (left power-law tail).
* `Argus(resonance, slope, power = 0.5)` = scipy `argus(chi, scale =
  resonance)` with slope = -chi^2 / 2, so chi = sqrt(40) here.

STATUS: live, on a two-repo change. flatppl-rust "determinizer: lower a
standard-module distribution member as a constructor" makes the
determiniser emit a §09 member as a bare-tag constructor; before it,
every §09 member refused with "cross-module ref could not be resolved
against the module bundle". flatppl-js "engine: accept a standard-module
member as a FlatPDL kernel tag" then lets that tag lower and stops the
analyzer reporting it as an undefined variable — the densities were
always REGISTRY-resident, so only the name was ever rejected.
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

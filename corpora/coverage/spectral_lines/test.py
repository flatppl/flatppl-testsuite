"""Independent oracle for cov_spectral_lines (scipy, never the engine).

The posterior is `bayesupdate(L, prior)`, so its log-density at
(w, sigma) is the sum of the prior terms and the likelihood:

    log Dirichlet(w | alpha) + log Gamma(sigma | shape, rate)
    + sum_j logsumexp_i(log w_i + log Normal(y_j | centers_i, sigma))

The mixture term is §06's `ksuperpose` density rule verbatim. The
Dirichlet term uses scipy's chart-measure formula, which the settled
§03/§06/§08 ruling makes normative; note that agreement on THAT term
alone is same-source (engine and oracle both transcribe §08) — the
mixture and Gamma terms are the independently checkable part.
`normalize` in the model contributes nothing: Σw = 1 exactly.
"""
import numpy as np
from scipy import stats
from scipy.special import logsumexp

CENTERS = np.array([412.1, 415.7, 421.3])
ALPHA = [2.0, 2.0, 2.0]
GAMMA_SHAPE, GAMMA_RATE = 3.0, 4.0

# Same values as model.flatppl's y_obs (numpy seed 41213 at
# w = (0.5, 0.3, 0.2), sigma = 0.9).
Y_OBS = np.array([
    416.9659, 412.0515, 421.551, 411.9125, 410.2642, 412.8685,
    410.7146, 421.7806, 420.9414, 411.5065, 420.6732, 420.4982,
    413.3433, 411.403, 412.5639, 411.3898, 412.7373, 421.1254,
    416.742, 421.9416, 412.5557, 415.002, 412.2671, 422.4918,
])


def oracle(point: dict) -> float:
    w = np.asarray(point["w"], dtype=float)
    sigma = float(point["sigma"])
    lp = stats.dirichlet.logpdf(w, ALPHA)
    lp += stats.gamma.logpdf(sigma, a=GAMMA_SHAPE, scale=1.0 / GAMMA_RATE)
    comp = np.log(w)[None, :] + stats.norm.logpdf(
        Y_OBS[:, None], CENTERS[None, :], sigma
    )
    lp += logsumexp(comp, axis=1).sum()
    return float(lp)

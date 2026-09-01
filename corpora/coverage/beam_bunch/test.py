"""Independent oracle for cov_beam_bunch (scipy).

Extended unbinned Poisson-process log-likelihood over the window R:

    sum_i log lambda(x_i) - Lambda(R) + priors
    lambda(x) = s * truncnorm.pdf(x; 5, 0.8 on R) + b * 0.1
    Lambda(R) = s + b  (both shapes have unit mass on R)

STATUS: the rust determiniser REFUSES this model — `PoissonProcess`
has no type rule ("its calls are left %deferred") and the `normalize`
node then has no determinization rule (verified live 2026-09-01). The
flatppl-js materialise route supports superpose intensities in
mat-poisson, so this pin is determiniser-side. `allow_skip: true`; the
frozen values are real oracle values and take over when the lowering
lands.
"""
import numpy as np
from scipy import stats

EVENTS = np.array([
    1.4558, 2.7229, 3.44, 3.5532, 3.815, 4.1881, 4.5518, 4.6498,
    4.7677, 4.8254, 4.8584, 4.9804, 5.0494, 5.2282, 5.4442, 5.5463,
    5.8047, 5.8668, 5.9805, 6.1686, 6.2184, 6.6791, 7.2659, 7.4487,
    9.375,
])
_A, _B = (0.0 - 5.0) / 0.8, (10.0 - 5.0) / 0.8


def oracle(point: dict) -> float:
    s, b = float(point["s"]), float(point["b"])
    lam = s * stats.truncnorm.pdf(EVENTS, _A, _B, loc=5.0, scale=0.8) + b * 0.1
    lik = np.log(lam).sum() - (s + b)
    prior = stats.gamma.logpdf(s, a=5.0, scale=2.0) + stats.gamma.logpdf(b, a=8.0, scale=2.0)
    return float(lik + prior)

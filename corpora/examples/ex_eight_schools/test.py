"""Independent oracle for ex_eight_schools.

Rubin's eight-schools model: ``mu ~ Normal(0,5)``, ``tau ~ half-Cauchy(0,5)``,
``theta ~ iid(Normal(mu, tau), 8)``, ``y | theta ~ Normal(theta, std_errs)``
against the fixed ``y_data``/``std_errs_data``. Half-Cauchy (``normalize(
truncate(Cauchy(0, 5), interval(0, inf)))``) contributes ``log(2) +
cauchy.logpdf(tau, 0, 5)`` for ``tau > 0`` (total mass of the positive half is
1/2 by symmetry about ``x0=0``).
"""
import math

from scipy.stats import cauchy, norm

_Y = [28, 8, -3, 7, -1, 1, 18, 12]
_SE = [15, 10, 16, 11, 9, 11, 10, 18]


def oracle(point: dict) -> float:
    mu, tau = point["mu"], point["tau"]
    theta = point["theta"]
    lp = norm.logpdf(mu, 0, 5) + (math.log(2) + cauchy.logpdf(tau, 0, 5))
    lp += sum(norm.logpdf(t, mu, tau) for t in theta)
    lp += sum(norm.logpdf(yi, ti, sei) for yi, ti, sei in zip(_Y, theta, _SE))
    return float(lp)

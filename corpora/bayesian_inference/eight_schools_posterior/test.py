"""Independent oracle for post_eight_schools.

Rubin's eight-schools model: ``mu ~ Normal(0,5)``, ``tau ~
half-Cauchy(0,5)`` (``normalize(truncate(Cauchy(0,5), interval(0,
inf)))`` doubles the Cauchy density on its positive half-line), ``theta ~
iid(Normal(mu, tau), 8)``, ``y | theta ~ Normal(theta, std_errs)``, scored
at ``mu=0.0, tau=1.0, theta=0^8`` against the fixed
``y_data``/``std_errs_data``.
"""
import math

from scipy.stats import cauchy, norm


def oracle() -> float:
    y = [28, 8, -3, 7, -1, 1, 18, 12]
    se = [15, 10, 16, 11, 9, 11, 10, 18]
    mu, tau = 0.0, 1.0
    theta = [0.0] * 8
    return (
        norm.logpdf(mu, 0, 5)
        + (math.log(2) + cauchy.logpdf(tau, 0, 5))
        + sum(norm.logpdf(th, mu, tau) for th in theta)
        + sum(norm.logpdf(yi, th, sei) for yi, th, sei in zip(y, theta, se))
    )

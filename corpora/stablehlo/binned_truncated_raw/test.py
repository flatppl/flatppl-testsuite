"""Gaussian CDF integrals give each restricted bin's Poisson mean.

The observed bins stop at 1, before the live restriction endpoint. Raw
restriction retains the original Gaussian masses without rescaling them;
moving that endpoint beyond the observed window has zero derivative.
"""

import math


def _interval(mu, sigma, lo, hi, moving=False):
    if hi <= lo:
        return 0.0, [0.0, 0.0, 0.0]
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    phi_a = math.exp(-a * a / 2) / math.sqrt(2 * math.pi)
    phi_b = math.exp(-b * b / 2) / math.sqrt(2 * math.pi)
    mass = (math.erfc(-b / math.sqrt(2)) - math.erfc(-a / math.sqrt(2))) / 2
    return mass, [
        (phi_a - phi_b) / sigma,
        (a * phi_a - b * phi_b) / sigma,
        phi_b / sigma if moving else 0.0,
    ]


def _answer(point):
    mu, sigma, edge = (point[name] for name in ("mu", "sigma", "edge"))
    total, dtotal = 1.0, [0.0, 0.0, 0.0]
    value, gradient = 0.0, [0.0, 0.0, 0.0]
    for lo, hi, count in ((-2, -1, 0), (-1, 0, 0), (0, 1, 2)):
        mass, dm = _interval(mu, sigma, max(lo, -0.5), min(hi, edge), hi > edge)
        rate = 5 * mass / total
        if rate == 0:
            assert count == 0
            continue
        value += count * math.log(rate) - rate - math.lgamma(count + 1)
        for j in range(3):
            dr = 5 * (dm[j] * total - mass * dtotal[j]) / total**2
            gradient[j] += (count / rate - 1) * dr
    return value, dict(zip(("mu", "sigma", "edge"), gradient))


def oracle(point):
    return _answer(point)[0]


def grad_oracle(point):
    return _answer(point)[1]

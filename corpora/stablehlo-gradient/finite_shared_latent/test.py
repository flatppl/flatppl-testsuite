"""A two-component bivariate Gaussian mixture with one shared Bernoulli draw."""

import math


def components(point):
    p, x, y = (point[key] for key in ("p", "px", "py"))
    a = math.exp(-0.5 * (x*x + y*y)) / (2*math.pi)
    b = math.exp(-0.5 * ((x-1)**2 + (y-2)**2)) / (2*math.pi)
    return p, x, y, a, b, (1-p)*a + p*b


def oracle(point):
    return math.log(components(point)[-1])


def grad_oracle(point):
    p, x, y, a, b, mass = components(point)
    return {"p": (b-a)/mass,
            "px": (-(1-p)*a*x + p*b*(1-x))/mass,
            "py": (-(1-p)*a*y + p*b*(2-y))/mass}

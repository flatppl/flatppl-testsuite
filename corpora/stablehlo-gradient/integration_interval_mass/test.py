"""Shape-2 Gamma survival difference and its analytic derivatives."""

import math


def log_mass(point):
    rate, lo, hi = point["rate"], max(point["lo"], 0.0), point["hi"]
    ratio = math.exp(-rate*(hi-lo))*(1+rate*hi)/(1+rate*lo)
    return -rate*lo + math.log1p(rate*lo) + math.log1p(-ratio)


def oracle(point):
    rate, x = point["rate"], point["point"]
    return 2*math.log(rate) + math.log(x) - rate*x - log_mass(point)


def grad_oracle(point):
    rate, lo, hi, x = (point[k] for k in ("rate", "lo", "hi", "point"))
    lower = max(lo, 0.0)
    z = log_mass(point)
    low_exp, high_exp = math.exp(-rate*lower-z), math.exp(-rate*hi-z)
    return {
        "rate": 2/rate-x + rate*(lower**2*low_exp-hi**2*high_exp),
        "lo": rate**2*lower*low_exp if lo > 0 else 0.0,
        "hi": -rate**2*hi*high_exp,
        "point": 1/x-rate,
    }

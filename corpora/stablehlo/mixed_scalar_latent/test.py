"""Analytic convolution of an atomic/Normal prior with a unit Normal kernel."""

import math


def components(point):
    p, atom, scale, x = (point[k] for k in ("p", "atom", "scale", "point"))
    variance = 1 + scale*scale
    a = math.log(p) - 0.5*(math.log(2*math.pi) + (x-atom)**2)
    b = math.log1p(-p) - 0.5*(math.log(2*math.pi*variance) + x*x/variance)
    shift = max(a, b)
    score = shift + math.log(math.exp(a-shift) + math.exp(b-shift))
    return score, math.exp(a-score), math.exp(b-score), variance


def oracle(point):
    return components(point)[0]


def grad_oracle(point):
    p, atom, scale, x = (point[k] for k in ("p", "atom", "scale", "point"))
    _, a, b, variance = components(point)
    return {"p": a/p - b/(1-p), "atom": a*(x-atom),
            "scale": b*scale*(x*x/variance**2 - 1/variance),
            "point": -a*(x-atom) - b*x/variance}

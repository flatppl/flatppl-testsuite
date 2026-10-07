"""Analytic derivative of the product and nested densities."""

import math


def grad_oracle(point):
    a = point["a"]
    z = math.expm1(a)/a
    dlogz = math.exp(a)/math.expm1(a) - 1/a
    density = math.exp(a*0.25)/z
    return {"a": 0.7 - 3*dlogz + density/(1+density)*(0.25-dlogz)}

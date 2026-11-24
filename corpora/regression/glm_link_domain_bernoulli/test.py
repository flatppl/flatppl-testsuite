"""Independent oracle: log density of `Bernoulli.(invlogit.(alpha .+ x * beta))` at y = [1, 0, 1].

The frozen values are mpmath at 50 digits, rounded to f64. This closed form
reproduces them in f64: a success scores log(invlogit(eta)) = -log1p(exp(-eta))
and a failure log(1 - invlogit(eta)) = -log1p(exp(eta)), each written so that
the exponent is never positive.
"""
import math

Y = [1, 0, 1]


def _log_invlogit(z: float) -> float:
    return -math.log1p(math.exp(-z)) if z >= 0 else z - math.log1p(math.exp(z))


def logdensity(x, alpha, beta) -> float:
    eta = [alpha + math.fsum(a * b for a, b in zip(row, beta)) for row in x]
    return math.fsum(_log_invlogit(m if y == 1 else -m) for y, m in zip(Y, eta))


def oracle(point: dict) -> float:
    return logdensity(point["x"], point["alpha"], point["beta"])

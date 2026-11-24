"""Independent oracle: log density of `Poisson.(exp.(alpha .+ x * beta))` at y = [0, 2, 3].

The frozen values are mpmath at 50 digits, rounded to f64. This closed form
reproduces them in f64: sum over rows of y eta - exp(eta) - lgamma(y + 1). At
alpha = 800 the exact value is about -8.18e347, below the f64 range, so the
oracle returns -inf once exp(eta) overflows.
"""
import math

Y = [0, 2, 3]


def logdensity(x, alpha, beta) -> float:
    eta = [alpha + math.fsum(a * b for a, b in zip(row, beta)) for row in x]
    if max(eta) > math.log(1.7976931348623157e308):
        return -math.inf
    return math.fsum(y * m - math.exp(m) - math.lgamma(y + 1) for y, m in zip(Y, eta))


def oracle(point: dict) -> float:
    return logdensity(point["x"], point["alpha"], point["beta"])

"""Independent oracle: log density of `Normal.(alpha .+ x * beta, sigma)` at y = [1.0, 2.0, 0.5].

The frozen values are mpmath at 50 digits, rounded to f64. This closed form
reproduces them in f64: sum over rows of -log(2 pi)/2 - log(sigma) - r^2/2 with
r = (y - eta) / sigma and eta = alpha + x beta.
"""
import math

Y = [1.0, 2.0, 0.5]


def logdensity(x, alpha, beta, sigma) -> float:
    eta = [alpha + math.fsum(a * b for a, b in zip(row, beta)) for row in x]
    return math.fsum(
        -0.5 * math.log(2.0 * math.pi) - math.log(sigma) - 0.5 * ((y - m) / sigma) ** 2
        for y, m in zip(Y, eta)
    )


def oracle(point: dict) -> float:
    return logdensity(point["x"], point["alpha"], point["beta"], point["sigma"])

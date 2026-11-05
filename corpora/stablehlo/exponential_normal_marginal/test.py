"""Independent scipy oracle: the exponentially modified Gaussian, the marginal of
y ~ Normal(mu = v, sigma = sig) under v ~ Exponential(rate = lam), at t_y."""
from scipy.stats import exponnorm


def oracle(point: dict) -> float:
    lam, sig = point["lam"], point["sig"]
    return float(exponnorm.logpdf(point["t_y"], 1.0 / (sig * lam), loc=0.0, scale=sig))

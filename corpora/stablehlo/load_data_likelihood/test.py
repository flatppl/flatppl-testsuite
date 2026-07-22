"""Independent scipy oracle: iid/broadcast Normal likelihood.
sum(norm.logpdf(y_data, loc=alpha + beta*x_data, scale=sigma))."""
import numpy as np
from scipy.stats import norm


def oracle(point: dict) -> float:
    alpha = point["alpha"]; beta = point["beta"]; sigma = point["sigma"]
    x = np.asarray(point["x_data"]); y = np.asarray(point["y_data"])
    means = alpha + beta * x
    return float(np.sum(norm.logpdf(y, loc=means, scale=sigma)))

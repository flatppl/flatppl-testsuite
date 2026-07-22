"""Independent scipy oracle: Dirichlet(alpha) logdensity at the fixed
observed a=[0.2, 0.3, 0.5]."""
from scipy.stats import dirichlet


def oracle(point: dict) -> float:
    alpha = point["alpha"]
    return float(dirichlet.logpdf([0.2, 0.3, 0.5], alpha))

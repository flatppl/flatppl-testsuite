"""Independent analytic oracle: gradient of Gamma(shape, rate) logdensity at
xobs w.r.t. shape and rate. logpdf = shape*log(rate) - lgamma(shape) +
(shape-1)*log(x) - rate*x, so d/dshape = log(rate) - digamma(shape) + log(x)
and d/drate = shape/rate - x."""
from scipy.special import digamma


def grad_oracle(point: dict) -> dict:
    shape = point["shape"]
    rate = point["rate"]
    x = point["xobs"]
    import math
    dshape = math.log(rate) - digamma(shape) + math.log(x)
    drate = shape / rate - x
    return {"shape": dshape, "rate": drate}

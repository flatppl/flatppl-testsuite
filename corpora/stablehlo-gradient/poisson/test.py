"""Independent analytic oracle: gradient of Poisson(rate) logdensity at the
fixed variate k=3 (see query.flatppl) w.r.t. rate. logpdf = k*log(rate) - rate
- lgamma(k+1), so d/drate = k/rate - 1."""

_K = 3


def grad_oracle(point: dict) -> dict:
    rate = point["rate"]
    return {"rate": _K / rate - 1.0}

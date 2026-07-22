"""Independent analytic oracle: gradient of Geometric(p) logdensity at the
fixed variate k=3 failures (see query.flatppl) w.r.t. p. logpdf =
k*log(1-p) + log(p), so d/dp = 1/p - k/(1-p)."""

_K = 3


def grad_oracle(point: dict) -> dict:
    p = point["p"]
    return {"p": 1.0 / p - _K / (1.0 - p)}

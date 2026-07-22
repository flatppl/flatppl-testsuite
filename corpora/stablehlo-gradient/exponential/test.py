"""Independent analytic oracle: gradient of Exponential(rate) logdensity at
xobs w.r.t. rate. logpdf = log(rate) - rate*x, so d/drate = 1/rate - x."""


def grad_oracle(point: dict) -> dict:
    rate = point["rate"]
    x = point["xobs"]
    return {"rate": 1.0 / rate - x}

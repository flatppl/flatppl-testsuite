"""Independent analytic oracle: gradient of Bernoulli(p) logdensity at the
fixed variate k=1 (see query.flatppl) w.r.t. p. logpdf = k*log(p) +
(1-k)*log(1-p), so at k=1, d/dp = 1/p."""


def grad_oracle(point: dict) -> dict:
    p = point["p"]
    return {"p": 1.0 / p}

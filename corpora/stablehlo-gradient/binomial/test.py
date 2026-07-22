"""Independent analytic oracle: gradient of Binomial(n, p) logdensity at the
fixed variate k=2 (see query.flatppl) w.r.t. p (n is data, not differentiated).
logpdf = lchoose(n,k) + k*log(p) + (n-k)*log(1-p), so
d/dp = k/p - (n-k)/(1-p)."""

_K = 2


def grad_oracle(point: dict) -> dict:
    n = point["n"]
    p = point["p"]
    return {"p": _K / p - (n - _K) / (1.0 - p)}

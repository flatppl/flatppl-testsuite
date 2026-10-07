"""The interior normalizer is 2-cut, so d(-log Z)/dcut = 1/(2-cut)."""


def grad_oracle(point):
    cut = point["cut"]
    return {"cut": 1/(2-cut) if 0 < cut < 1 else 0.0}

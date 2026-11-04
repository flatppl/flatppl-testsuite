"""Horner's rule: value = sum_i c[i] p^(n-1-i), differentiated term by term."""


def grad_oracle(point):
    p, c = point["p"], point["coefficients"]
    n = len(c)
    return {"p": sum(c[i]*(n-1-i)*p**(n-2-i) for i in range(n - 1)),
            "coefficients": [p**(n-1-i) for i in range(n)]}

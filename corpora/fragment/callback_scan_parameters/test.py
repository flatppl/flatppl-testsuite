"""Expand the three recurrence steps, passing the same tuple at each step."""


def oracle(point):
    a, b = point["a"], point["b"]
    return a*a*(1+b) + a*(2+b) + 3+b

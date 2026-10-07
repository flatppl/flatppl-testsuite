"""Categorical0 uses zero-based support and zero mass outside it."""
import math

def oracle(point):
    p, i = point["p"], point["observed"]
    return math.log(p[i]) if 0 <= i < len(p) and p[i] > 0 else -math.inf

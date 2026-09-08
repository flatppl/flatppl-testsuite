"""A Boolean ABI argument selects two distinct Normal densities."""
import math


def oracle(point):
    center = 2 if point["active"] else -3
    return -0.5 * math.log(2 * math.pi) - center**2 / 2

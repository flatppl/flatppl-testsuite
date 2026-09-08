"""Integer arrays and real scalars follow their ABI domains, not JSON spelling."""
import math


def oracle(point):
    center = point["mu"] + sum(point["v"])
    return -0.5 * math.log(2 * math.pi) - math.log(2) - (5 - center)**2 / 8

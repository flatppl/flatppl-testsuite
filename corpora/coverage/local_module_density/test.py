"""A local module exports Normal(2,3); score its law with an explicit formula."""
import math


def oracle(point):
    return -0.5 * math.log(2 * math.pi) - math.log(3) - (point["x"] - 2)**2 / 18

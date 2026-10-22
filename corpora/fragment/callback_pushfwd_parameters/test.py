"""The affine pushforward is Normal(shift, scale), by change of variables."""

import math


def oracle(point):
    scale, shift = point["scale"], point["shift"]
    return -math.log(2*math.pi)/2 - math.log(scale) - (2-shift)**2/(2*scale**2)

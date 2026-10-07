"""The uniform background has mass one; the narrow plateau adds 100*(hi-lo)."""

import math


def oracle(point):
    return -math.log1p(100*(point["hi"]-point["lo"]))

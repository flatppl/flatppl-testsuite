"""§07 Field and element access: A[:,2] selects column 2, while get0
counts both indices from zero. Thus mu = v[1] + 16 + 32 in Python's
indexing convention. The explicit Normal formula supplies the oracle."""
import math


def oracle(point: dict) -> float:
    mu = point["v"][1] + 48.0
    return -0.5 * math.log(2 * math.pi) - math.log(2) - 0.5 * ((51 - mu) / 2) ** 2
